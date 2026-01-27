#!/usr/bin/env python3
"""
pipeline_diversity_engine.py

Pipeline Diversity Scoring Engine for Biotech Screener

Analyzes pipeline breadth, phase diversity, and platform characteristics
to assess company risk profile. Diversified pipelines have lower binary
risk than single-asset companies.

Key Metrics:
- Pipeline Depth: Number of distinct clinical programs
- Phase Diversity: Distribution of programs across development phases
- Indication Diversity: Spread across therapeutic areas
- Platform Score: Technology platform validation (multiple programs from same platform)

Risk Implications:
- Single-asset companies: Higher binary risk, but potentially higher return
- Platform companies: Lower binary risk, platform failure is catastrophic
- Diversified portfolios: Balanced risk profile

Author: Wake Robin Capital Management
Version: 1.0.0
"""

import hashlib
import json
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import date
from collections import Counter
from dataclasses import dataclass
from enum import Enum


__version__ = "1.0.0"
__author__ = "Wake Robin Capital Management"


class PipelineRiskProfile(Enum):
    """Pipeline risk classification."""
    SINGLE_ASSET = "single_asset"           # 1 program, highest binary risk
    CONCENTRATED = "concentrated"            # 2-3 programs, high binary risk
    FOCUSED = "focused"                      # 4-6 programs, moderate risk
    DIVERSIFIED = "diversified"             # 7-12 programs, lower risk
    BROAD_PORTFOLIO = "broad_portfolio"     # 13+ programs, institutional profile


class PlatformType(Enum):
    """Technology platform categories."""
    SMALL_MOLECULE = "small_molecule"
    ANTIBODY = "antibody"
    BISPECIFIC = "bispecific"
    ADC = "adc"                    # Antibody-drug conjugate
    CAR_T = "car_t"
    GENE_THERAPY = "gene_therapy"
    RNA = "rna"                    # mRNA, siRNA, ASO
    CELL_THERAPY = "cell_therapy"
    PROTEIN = "protein"
    VACCINE = "vaccine"
    UNKNOWN = "unknown"


@dataclass
class PipelineProgram:
    """Individual program in a company's pipeline."""
    ticker: str
    program_name: Optional[str]
    nct_id: Optional[str]
    phase: str
    indication: str
    platform_type: PlatformType
    is_lead_program: bool


class PipelineDiversityEngine:
    """
    Pipeline diversity scoring for biotech risk assessment.

    Evaluates pipeline characteristics to assess company risk profile:
    - Single-asset companies have binary (all-or-nothing) outcomes
    - Platform companies can leverage technology across multiple programs
    - Diversified portfolios reduce idiosyncratic program risk

    Usage:
        engine = PipelineDiversityEngine()
        result = engine.score_pipeline(
            ticker="ACME",
            trial_records=trials,
            as_of_date=date(2026, 1, 26)
        )
    """

    VERSION = "1.0.0"

    # Phase progression for diversity calculation
    PHASE_ORDER = ["preclinical", "phase_1", "phase_1_2", "phase_2", "phase_2_3", "phase_3", "nda_bla", "commercial"]
    PHASE_WEIGHTS = {
        "preclinical": Decimal("0.5"),
        "phase_1": Decimal("1.0"),
        "phase_1_2": Decimal("1.5"),
        "phase_2": Decimal("2.0"),
        "phase_2_3": Decimal("2.5"),
        "phase_3": Decimal("3.0"),
        "nda_bla": Decimal("3.5"),
        "commercial": Decimal("4.0"),
    }

    # Risk profile thresholds (by weighted program count)
    RISK_PROFILE_THRESHOLDS = {
        PipelineRiskProfile.SINGLE_ASSET: (Decimal("0"), Decimal("1.5")),
        PipelineRiskProfile.CONCENTRATED: (Decimal("1.5"), Decimal("4")),
        PipelineRiskProfile.FOCUSED: (Decimal("4"), Decimal("10")),
        PipelineRiskProfile.DIVERSIFIED: (Decimal("10"), Decimal("25")),
        PipelineRiskProfile.BROAD_PORTFOLIO: (Decimal("25"), Decimal("999")),
    }

    # Score adjustments by risk profile
    RISK_PROFILE_SCORE_ADJUSTMENTS: Dict[PipelineRiskProfile, Decimal] = {
        PipelineRiskProfile.SINGLE_ASSET: Decimal("-15"),      # Higher risk, lower score
        PipelineRiskProfile.CONCENTRATED: Decimal("-8"),
        PipelineRiskProfile.FOCUSED: Decimal("0"),             # Baseline
        PipelineRiskProfile.DIVERSIFIED: Decimal("8"),
        PipelineRiskProfile.BROAD_PORTFOLIO: Decimal("12"),    # Institutional profile
    }

    # Platform validation bonus (having multiple programs proves platform works)
    PLATFORM_VALIDATION_THRESHOLD = 3  # Programs needed to validate platform
    PLATFORM_VALIDATION_BONUS = Decimal("10")

    # Indication diversity bonus
    INDICATION_DIVERSITY_THRESHOLDS = {
        1: Decimal("0"),      # Single indication
        2: Decimal("3"),      # Two indications
        3: Decimal("6"),      # Three indications
        4: Decimal("8"),      # Four+ indications
    }

    # Phase diversity bonus (having programs across multiple phases)
    PHASE_DIVERSITY_THRESHOLDS = {
        1: Decimal("0"),      # Single phase
        2: Decimal("4"),      # Two phases
        3: Decimal("8"),      # Three phases
        4: Decimal("10"),     # Four+ phases
    }

    def __init__(self):
        """Initialize the pipeline diversity engine."""
        self.audit_trail: List[Dict[str, Any]] = []

    def score_pipeline(
        self,
        ticker: str,
        trial_records: List[Dict[str, Any]],
        as_of_date: date,
        clinical_score_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Calculate pipeline diversity score for a ticker.

        Args:
            ticker: Stock ticker
            trial_records: List of clinical trial records
            as_of_date: Point-in-time date
            clinical_score_data: Optional clinical scoring data with lead_phase, etc.

        Returns:
            Dict containing:
            - diversity_score: 0-100 pipeline diversity score
            - risk_profile: PipelineRiskProfile classification
            - program_count: Total distinct programs
            - phase_distribution: Programs by phase
            - indication_distribution: Programs by indication
            - platform_validated: Whether platform has multiple programs
            - audit_entry: Full calculation trace
        """
        ticker = ticker.upper()

        # Extract programs for this ticker
        programs = self._extract_programs(ticker, trial_records, as_of_date)

        if not programs:
            # Fall back to clinical_score_data if available
            if clinical_score_data:
                programs = self._programs_from_clinical_data(ticker, clinical_score_data)

        if not programs:
            return self._empty_pipeline_result(ticker, as_of_date)

        # Calculate metrics
        program_count = len(programs)
        weighted_count = self._calculate_weighted_program_count(programs)
        phase_distribution = self._calculate_phase_distribution(programs)
        indication_distribution = self._calculate_indication_distribution(programs)
        platform_distribution = self._calculate_platform_distribution(programs)

        # Determine risk profile
        risk_profile = self._classify_risk_profile(weighted_count, program_count)

        # Calculate diversity score components
        base_score = Decimal("50")  # Neutral starting point

        # 1. Risk profile adjustment
        risk_adjustment = self.RISK_PROFILE_SCORE_ADJUSTMENTS[risk_profile]

        # 2. Phase diversity bonus
        phase_count = len([p for p, c in phase_distribution.items() if c > 0])
        phase_diversity_bonus = self.PHASE_DIVERSITY_THRESHOLDS.get(
            min(phase_count, 4), Decimal("10")
        )

        # 3. Indication diversity bonus
        indication_count = len([i for i, c in indication_distribution.items() if c > 0])
        indication_diversity_bonus = self.INDICATION_DIVERSITY_THRESHOLDS.get(
            min(indication_count, 4), Decimal("8")
        )

        # 4. Platform validation bonus
        platform_validated = False
        platform_bonus = Decimal("0")
        for platform, count in platform_distribution.items():
            if count >= self.PLATFORM_VALIDATION_THRESHOLD:
                platform_validated = True
                platform_bonus = self.PLATFORM_VALIDATION_BONUS
                break

        # Calculate final score
        diversity_score = (
            base_score
            + risk_adjustment
            + phase_diversity_bonus
            + indication_diversity_bonus
            + platform_bonus
        )

        # Clamp to 0-100
        diversity_score = max(Decimal("0"), min(Decimal("100"), diversity_score))
        diversity_score = diversity_score.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)

        # Build audit entry
        audit_entry = {
            "ticker": ticker,
            "as_of_date": as_of_date.isoformat(),
            "program_count": program_count,
            "weighted_program_count": str(weighted_count),
            "risk_profile": risk_profile.value,
            "phase_distribution": phase_distribution,
            "indication_distribution": indication_distribution,
            "platform_distribution": {k.value: v for k, v in platform_distribution.items()},
            "score_components": {
                "base_score": str(base_score),
                "risk_adjustment": str(risk_adjustment),
                "phase_diversity_bonus": str(phase_diversity_bonus),
                "indication_diversity_bonus": str(indication_diversity_bonus),
                "platform_bonus": str(platform_bonus),
                "final_score": str(diversity_score),
            },
            "platform_validated": platform_validated,
        }

        self.audit_trail.append(audit_entry)

        return {
            "ticker": ticker,
            "diversity_score": diversity_score,
            "risk_profile": risk_profile.value,
            "risk_profile_description": self._risk_profile_description(risk_profile),
            "program_count": program_count,
            "weighted_program_count": weighted_count,
            "phase_distribution": phase_distribution,
            "phase_diversity_count": phase_count,
            "indication_distribution": indication_distribution,
            "indication_diversity_count": indication_count,
            "platform_validated": platform_validated,
            "dominant_platform": self._get_dominant_platform(platform_distribution),
            "audit_entry": audit_entry,
        }

    def _extract_programs(
        self,
        ticker: str,
        trial_records: List[Dict[str, Any]],
        as_of_date: date
    ) -> List[PipelineProgram]:
        """Extract distinct programs from trial records."""
        programs = []
        seen_nct_ids: Set[str] = set()

        for trial in trial_records:
            trial_ticker = (
                trial.get("lead_sponsor_ticker") or
                trial.get("ticker") or ""
            ).upper()

            if trial_ticker != ticker:
                continue

            nct_id = trial.get("nct_id", "")
            if nct_id in seen_nct_ids:
                continue
            seen_nct_ids.add(nct_id)

            # Filter by as_of_date if we have date info
            start_date = trial.get("start_date") or trial.get("primary_completion_date")
            # Skip future trials
            # (simplified - in production would parse dates properly)

            phase = self._normalize_phase(trial.get("phase", ""))
            indication = self._extract_indication(trial.get("conditions", []))
            platform = self._infer_platform(trial.get("interventions", []))

            programs.append(PipelineProgram(
                ticker=ticker,
                program_name=trial.get("title", "")[:50] if trial.get("title") else None,
                nct_id=nct_id,
                phase=phase,
                indication=indication,
                platform_type=platform,
                is_lead_program=False,
            ))

        return programs

    def _programs_from_clinical_data(
        self,
        ticker: str,
        clinical_data: Dict[str, Any]
    ) -> List[PipelineProgram]:
        """Create synthetic programs from clinical score data."""
        programs = []

        n_trials = clinical_data.get("n_trials_unique", 0)
        lead_phase = clinical_data.get("lead_phase", "phase_2")

        if n_trials == 0:
            return programs

        # Create one representative program per estimated phase
        phase_normalized = self._normalize_phase(lead_phase)

        # Estimate phase distribution based on trial count
        # This is a heuristic - in production would use actual data
        if n_trials >= 50:
            # Large pipeline - likely multiple phases
            estimated_phases = ["phase_1", "phase_2", "phase_3"]
            programs_per_phase = n_trials // len(estimated_phases)
        elif n_trials >= 20:
            estimated_phases = ["phase_1", "phase_2"]
            programs_per_phase = n_trials // len(estimated_phases)
        else:
            estimated_phases = [phase_normalized]
            programs_per_phase = n_trials

        for phase in estimated_phases:
            for i in range(min(programs_per_phase, 10)):  # Cap at 10 per phase
                programs.append(PipelineProgram(
                    ticker=ticker,
                    program_name=f"Program_{phase}_{i+1}",
                    nct_id=None,
                    phase=phase,
                    indication="unknown",
                    platform_type=PlatformType.UNKNOWN,
                    is_lead_program=(i == 0 and phase == phase_normalized),
                ))

        return programs

    def _normalize_phase(self, phase: str) -> str:
        """Normalize phase string to standard format."""
        if not phase:
            return "phase_2"

        phase_lower = phase.lower().strip()

        phase_map = {
            "preclinical": "preclinical",
            "pre-clinical": "preclinical",
            "phase 1": "phase_1",
            "phase1": "phase_1",
            "phase i": "phase_1",
            "phase_1": "phase_1",
            "phase 1/2": "phase_1_2",
            "phase1/2": "phase_1_2",
            "phase 2": "phase_2",
            "phase2": "phase_2",
            "phase ii": "phase_2",
            "phase_2": "phase_2",
            "phase 2/3": "phase_2_3",
            "phase 3": "phase_3",
            "phase3": "phase_3",
            "phase iii": "phase_3",
            "phase_3": "phase_3",
            "nda": "nda_bla",
            "bla": "nda_bla",
            "approved": "commercial",
            "commercial": "commercial",
            "marketed": "commercial",
        }

        return phase_map.get(phase_lower, "phase_2")

    def _extract_indication(self, conditions: Any) -> str:
        """Extract primary indication from conditions."""
        if not conditions:
            return "unknown"

        if isinstance(conditions, str):
            conditions = [conditions]

        if not conditions:
            return "unknown"

        # Simple keyword matching for indication category
        condition_text = " ".join(conditions).lower()

        indication_keywords = {
            "oncology": ["cancer", "tumor", "carcinoma", "leukemia", "lymphoma", "melanoma", "oncolog"],
            "rare_disease": ["rare", "orphan"],
            "neurology": ["alzheimer", "parkinson", "multiple sclerosis", "epilepsy", "neurolog", "cns"],
            "infectious_disease": ["infection", "viral", "bacterial", "hiv", "hepatitis", "covid"],
            "cardiovascular": ["cardiovascular", "heart", "cardiac", "hypertension"],
            "immunology": ["autoimmune", "rheumatoid", "lupus", "psoriasis", "immunolog"],
            "metabolic": ["diabetes", "obesity", "metabolic"],
            "respiratory": ["respiratory", "copd", "asthma", "pulmonary"],
        }

        for indication, keywords in indication_keywords.items():
            if any(kw in condition_text for kw in keywords):
                return indication

        return "other"

    def _infer_platform(self, interventions: Any) -> PlatformType:
        """Infer technology platform from intervention descriptions."""
        if not interventions:
            return PlatformType.UNKNOWN

        if isinstance(interventions, str):
            interventions = [interventions]

        intervention_text = " ".join(str(i) for i in interventions).lower()

        platform_keywords = {
            PlatformType.CAR_T: ["car-t", "car t", "chimeric antigen"],
            PlatformType.GENE_THERAPY: ["gene therapy", "aav", "lentiviral", "crispr"],
            PlatformType.RNA: ["mrna", "sirna", "antisense", "aso", "rna"],
            PlatformType.ADC: ["adc", "antibody-drug conjugate", "antibody drug conjugate"],
            PlatformType.BISPECIFIC: ["bispecific", "bi-specific"],
            PlatformType.ANTIBODY: ["antibody", "mab", "monoclonal"],
            PlatformType.CELL_THERAPY: ["cell therapy", "stem cell"],
            PlatformType.VACCINE: ["vaccine", "immunization"],
            PlatformType.PROTEIN: ["protein", "enzyme", "fusion"],
        }

        for platform, keywords in platform_keywords.items():
            if any(kw in intervention_text for kw in keywords):
                return platform

        return PlatformType.SMALL_MOLECULE

    def _calculate_weighted_program_count(self, programs: List[PipelineProgram]) -> Decimal:
        """Calculate phase-weighted program count."""
        total = Decimal("0")
        for program in programs:
            weight = self.PHASE_WEIGHTS.get(program.phase, Decimal("1.0"))
            total += weight
        return total.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)

    def _calculate_phase_distribution(self, programs: List[PipelineProgram]) -> Dict[str, int]:
        """Calculate distribution of programs by phase."""
        distribution = {phase: 0 for phase in self.PHASE_ORDER}
        for program in programs:
            if program.phase in distribution:
                distribution[program.phase] += 1
        return distribution

    def _calculate_indication_distribution(self, programs: List[PipelineProgram]) -> Dict[str, int]:
        """Calculate distribution of programs by indication."""
        counter = Counter(p.indication for p in programs)
        return dict(counter)

    def _calculate_platform_distribution(self, programs: List[PipelineProgram]) -> Dict[PlatformType, int]:
        """Calculate distribution of programs by platform."""
        counter = Counter(p.platform_type for p in programs)
        return dict(counter)

    def _classify_risk_profile(
        self,
        weighted_count: Decimal,
        raw_count: int = 0
    ) -> PipelineRiskProfile:
        """
        Classify risk profile based on weighted program count.

        A single-program company is always SINGLE_ASSET regardless of phase,
        since company fate depends on one program.
        """
        # Single program is always single-asset (highest binary risk)
        if raw_count == 1:
            return PipelineRiskProfile.SINGLE_ASSET

        for profile, (min_val, max_val) in self.RISK_PROFILE_THRESHOLDS.items():
            if min_val <= weighted_count < max_val:
                return profile
        return PipelineRiskProfile.BROAD_PORTFOLIO

    def _risk_profile_description(self, profile: PipelineRiskProfile) -> str:
        """Get description for risk profile."""
        descriptions = {
            PipelineRiskProfile.SINGLE_ASSET: "High binary risk - single program determines company fate",
            PipelineRiskProfile.CONCENTRATED: "Elevated binary risk - limited pipeline diversification",
            PipelineRiskProfile.FOCUSED: "Balanced risk - focused pipeline with some diversification",
            PipelineRiskProfile.DIVERSIFIED: "Lower risk - diversified pipeline reduces program-specific risk",
            PipelineRiskProfile.BROAD_PORTFOLIO: "Institutional profile - broad portfolio like large biotech/pharma",
        }
        return descriptions.get(profile, "Unknown risk profile")

    def _get_dominant_platform(self, platform_distribution: Dict[PlatformType, int]) -> Optional[str]:
        """Get the dominant technology platform."""
        if not platform_distribution:
            return None
        dominant = max(platform_distribution.items(), key=lambda x: x[1])
        return dominant[0].value if dominant[1] > 0 else None

    def _empty_pipeline_result(self, ticker: str, as_of_date: date) -> Dict[str, Any]:
        """Return result for ticker with no pipeline data."""
        return {
            "ticker": ticker,
            "diversity_score": Decimal("35"),  # Below average for unknown
            "risk_profile": PipelineRiskProfile.SINGLE_ASSET.value,
            "risk_profile_description": "Unknown pipeline - assuming single-asset risk",
            "program_count": 0,
            "weighted_program_count": Decimal("0"),
            "phase_distribution": {},
            "phase_diversity_count": 0,
            "indication_distribution": {},
            "indication_diversity_count": 0,
            "platform_validated": False,
            "dominant_platform": None,
            "audit_entry": {
                "ticker": ticker,
                "as_of_date": as_of_date.isoformat(),
                "note": "no_pipeline_data",
            },
        }

    def score_universe(
        self,
        universe: List[Dict[str, Any]],
        trial_records: List[Dict[str, Any]],
        as_of_date: date
    ) -> Dict[str, Any]:
        """
        Score pipeline diversity for an entire universe.

        Args:
            universe: List of dicts with 'ticker' and optional clinical data
            trial_records: All trial records for the universe
            as_of_date: Point-in-time date

        Returns:
            Dict with scores, diagnostics, and provenance
        """
        scores = []
        risk_distribution: Dict[str, int] = {}

        for company in universe:
            ticker = company.get("ticker", "").upper()
            clinical_data = company.get("clinical_data", {})

            result = self.score_pipeline(
                ticker=ticker,
                trial_records=trial_records,
                as_of_date=as_of_date,
                clinical_score_data=clinical_data,
            )

            scores.append({
                "ticker": ticker,
                "diversity_score": result["diversity_score"],
                "risk_profile": result["risk_profile"],
                "program_count": result["program_count"],
                "phase_diversity_count": result["phase_diversity_count"],
                "indication_diversity_count": result["indication_diversity_count"],
                "platform_validated": result["platform_validated"],
            })

            profile = result["risk_profile"]
            risk_distribution[profile] = risk_distribution.get(profile, 0) + 1

        # Content hash
        scores_json = json.dumps(
            [{"t": s["ticker"], "s": str(s["diversity_score"])} for s in scores],
            sort_keys=True
        )
        content_hash = hashlib.sha256(scores_json.encode()).hexdigest()[:16]

        return {
            "as_of_date": as_of_date.isoformat(),
            "scores": scores,
            "diagnostic_counts": {
                "total_scored": len(scores),
                "risk_distribution": risk_distribution,
                "platform_validated_count": sum(1 for s in scores if s["platform_validated"]),
                "avg_program_count": sum(s["program_count"] for s in scores) / max(1, len(scores)),
            },
            "provenance": {
                "module": "pipeline_diversity_engine",
                "module_version": self.VERSION,
                "content_hash": content_hash,
                "pit_cutoff": as_of_date.isoformat(),
            },
        }

    def get_audit_trail(self) -> List[Dict[str, Any]]:
        """Return the audit trail."""
        return self.audit_trail.copy()

    def clear_audit_trail(self) -> None:
        """Clear the audit trail."""
        self.audit_trail = []


if __name__ == "__main__":
    # Demonstration
    print("=" * 70)
    print("PIPELINE DIVERSITY ENGINE v1.0.0 - DEMONSTRATION")
    print("=" * 70)

    engine = PipelineDiversityEngine()

    # Create sample clinical data
    sample_companies = [
        {"ticker": "SINGLE", "clinical_data": {"n_trials_unique": 1, "lead_phase": "phase_2"}},
        {"ticker": "FOCUSED", "clinical_data": {"n_trials_unique": 8, "lead_phase": "phase_3"}},
        {"ticker": "DIVERSE", "clinical_data": {"n_trials_unique": 50, "lead_phase": "commercial"}},
        {"ticker": "MEGA", "clinical_data": {"n_trials_unique": 200, "lead_phase": "commercial"}},
    ]

    as_of = date(2026, 1, 26)

    print(f"\nScoring pipelines as of {as_of}:")
    print("-" * 70)

    for company in sample_companies:
        result = engine.score_pipeline(
            ticker=company["ticker"],
            trial_records=[],  # Using clinical_data instead
            as_of_date=as_of,
            clinical_score_data=company["clinical_data"],
        )
        print(f"\n{company['ticker']}:")
        print(f"  Diversity Score: {result['diversity_score']}")
        print(f"  Risk Profile: {result['risk_profile']}")
        print(f"  Program Count: {result['program_count']}")
        print(f"  Phase Diversity: {result['phase_diversity_count']} phases")
        print(f"  Platform Validated: {result['platform_validated']}")
