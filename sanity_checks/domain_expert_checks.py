"""
Query 7.4: Biotech Domain Expert Checks

Implements biotech-specific business logic that catches domain errors:

1. Clinical Trial Reality Checks
2. Market Opportunity Checks
3. Regulatory Path Validation
4. Partnership/Platform Checks

These are the domain-specific validations that require biotech industry
knowledge to catch errors in the screening output.

Author: Wake Robin Capital Management
Version: 1.0.0
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set

from sanity_checks.types import (
    CheckCategory,
    FlagSeverity,
    SanityCheckResult,
    SanityFlag,
    SecurityContext,
    ThresholdConfig,
    DEFAULT_THRESHOLDS,
)

logger = logging.getLogger(__name__)


@dataclass
class TrialDetails:
    """Detailed trial information for domain validation."""
    ticker: str
    nct_id: str
    phase: str
    indication: str
    primary_endpoint: str
    enrollment: int
    has_orphan_designation: bool = False
    has_breakthrough_designation: bool = False
    has_fast_track: bool = False
    has_accelerated_approval_path: bool = False
    is_survival_endpoint: bool = False


@dataclass
class PartnershipInfo:
    """Partnership information for validation."""
    ticker: str
    partner_name: str
    partner_tier: str  # "major_pharma", "mid_size", "biotech"
    deal_value_mm: Optional[Decimal]
    milestone_payments: Optional[Decimal]
    royalty_range: Optional[str]


@dataclass
class CompetitiveLandscape:
    """Competitive landscape for an indication."""
    indication: str
    approved_therapies: int
    phase3_competitors: int
    differentiation_factors: List[str]


class DomainExpertChecker:
    """
    Biotech domain expert checker.

    Validates outputs against biotech industry knowledge and
    regulatory requirements.
    """

    # Minimum enrollment thresholds by phase and indication type
    MIN_ENROLLMENT = {
        "oncology_phase3": 100,
        "rare_disease_phase3": 30,
        "rare_disease_phase2": 20,
        "default_phase3": 50,
        "default_phase2": 30,
    }

    # Ultra-rare disease threshold (patients)
    ULTRA_RARE_THRESHOLD = 5000

    # Major pharma partners
    MAJOR_PHARMA = {
        "pfizer", "roche", "novartis", "merck", "jnj", "johnson & johnson",
        "abbvie", "bms", "bristol-myers squibb", "astrazeneca", "sanofi",
        "gsk", "glaxosmithkline", "eli lilly", "lilly", "gilead",
        "regeneron", "amgen", "biogen", "vertex", "takeda", "daiichi",
    }

    def __init__(self, config: Optional[ThresholdConfig] = None) -> None:
        self.config = config or DEFAULT_THRESHOLDS

    def run_all_checks(
        self,
        securities: List[SecurityContext],
        trial_details: Optional[List[TrialDetails]] = None,
        partnerships: Optional[List[PartnershipInfo]] = None,
        competitive_data: Optional[Dict[str, CompetitiveLandscape]] = None,
    ) -> SanityCheckResult:
        """
        Run all domain expert checks.

        Args:
            securities: List of security contexts
            trial_details: Optional detailed trial information
            partnerships: Optional partnership information
            competitive_data: Optional competitive landscape by indication

        Returns:
            SanityCheckResult with all flags
        """
        flags: List[SanityFlag] = []

        # 1. Clinical Trial Reality Checks
        if trial_details:
            flags.extend(self._check_clinical_trial_reality(trial_details))

        # 2. Market Opportunity Checks
        flags.extend(self._check_market_opportunity(securities, competitive_data))

        # 3. Regulatory Path Validation
        if trial_details:
            flags.extend(self._check_regulatory_path(trial_details, securities))

        # 4. Partnership/Platform Checks
        if partnerships:
            flags.extend(self._check_partnerships(securities, partnerships))

        # 5. Stage Progression Logic
        flags.extend(self._check_stage_logic(securities))

        # Calculate metrics
        metrics = self._calculate_metrics(flags, securities)

        passed = not any(f.severity == FlagSeverity.CRITICAL for f in flags)

        return SanityCheckResult(
            check_name="domain_expert",
            category=CheckCategory.DOMAIN_EXPERT,
            passed=passed,
            flags=flags,
            metrics=metrics,
        )

    def _check_clinical_trial_reality(
        self,
        trials: List[TrialDetails],
    ) -> List[SanityFlag]:
        """
        Check clinical trial reality.

        Flags:
        - Underpowered trials
        - Survival endpoints in Phase 2 rare disease
        - Regulatory path issues
        """
        flags: List[SanityFlag] = []

        for trial in trials:
            # Check 1: Enrollment sufficiency
            is_oncology = self._is_oncology_indication(trial.indication)
            is_rare = self._is_rare_disease(trial.indication)

            if trial.phase == "Phase 3":
                if is_oncology and trial.enrollment < self.MIN_ENROLLMENT["oncology_phase3"]:
                    flags.append(SanityFlag(
                        severity=FlagSeverity.HIGH,
                        category=CheckCategory.DOMAIN_EXPERT,
                        ticker=trial.ticker,
                        check_name="underpowered_trial",
                        message=f"Phase 3 oncology trial with N={trial.enrollment} patients",
                        details={
                            "nct_id": trial.nct_id,
                            "indication": trial.indication,
                            "enrollment": trial.enrollment,
                            "minimum_expected": self.MIN_ENROLLMENT["oncology_phase3"],
                        },
                        recommendation="Underpowered trial for primary endpoint",
                    ))
                elif not is_rare and trial.enrollment < self.MIN_ENROLLMENT["default_phase3"]:
                    flags.append(SanityFlag(
                        severity=FlagSeverity.MEDIUM,
                        category=CheckCategory.DOMAIN_EXPERT,
                        ticker=trial.ticker,
                        check_name="small_phase3_enrollment",
                        message=f"Phase 3 trial with N={trial.enrollment} patients for {trial.indication}",
                        details={
                            "nct_id": trial.nct_id,
                            "indication": trial.indication,
                            "enrollment": trial.enrollment,
                        },
                        recommendation="Verify trial is adequately powered",
                    ))

            # Check 2: Survival endpoint in Phase 2 rare disease
            if trial.phase == "Phase 2" and is_rare:
                if trial.is_survival_endpoint:
                    flags.append(SanityFlag(
                        severity=FlagSeverity.MEDIUM,
                        category=CheckCategory.DOMAIN_EXPERT,
                        ticker=trial.ticker,
                        check_name="regulatory_path_risk",
                        message=f"Phase 2 rare disease with survival primary endpoint",
                        details={
                            "nct_id": trial.nct_id,
                            "indication": trial.indication,
                            "primary_endpoint": trial.primary_endpoint,
                        },
                        recommendation="Regulatory path risk - typically Phase 3 requirement for survival",
                    ))

            # Check 3: Accelerated approval without confirmatory plan visibility
            if trial.has_accelerated_approval_path:
                # Would need additional data to fully validate, flag for review
                flags.append(SanityFlag(
                    severity=FlagSeverity.LOW,
                    category=CheckCategory.DOMAIN_EXPERT,
                    ticker=trial.ticker,
                    check_name="accelerated_approval_review",
                    message=f"Accelerated approval path flagged - verify confirmatory trial plan",
                    details={
                        "nct_id": trial.nct_id,
                        "indication": trial.indication,
                    },
                    recommendation="Verify confirmatory trial plan exists - withdrawal risk",
                ))

        return flags

    def _check_market_opportunity(
        self,
        securities: List[SecurityContext],
        competitive_data: Optional[Dict[str, CompetitiveLandscape]],
    ) -> List[SanityFlag]:
        """
        Check market opportunity realism.

        Flags:
        - Ultra-rare without orphan designation
        - Crowded market without differentiation
        """
        flags: List[SanityFlag] = []

        for sec in securities:
            if not sec.indication:
                continue

            # Check competitive landscape if available
            if competitive_data and sec.indication in competitive_data:
                landscape = competitive_data[sec.indication]

                # Crowded market check
                if landscape.approved_therapies >= 3 and sec.rank is not None and sec.rank <= 20:
                    if not landscape.differentiation_factors:
                        flags.append(SanityFlag(
                            severity=FlagSeverity.MEDIUM,
                            category=CheckCategory.DOMAIN_EXPERT,
                            ticker=sec.ticker,
                            check_name="crowded_market_risk",
                            message=f"{landscape.approved_therapies} approved therapies in {sec.indication} but no differentiation score",
                            details={
                                "indication": sec.indication,
                                "approved_therapies": landscape.approved_therapies,
                                "phase3_competitors": landscape.phase3_competitors,
                                "rank": sec.rank,
                            },
                            recommendation="Crowded market risk not reflected",
                        ))

        return flags

    def _check_regulatory_path(
        self,
        trials: List[TrialDetails],
        securities: List[SecurityContext],
    ) -> List[SanityFlag]:
        """
        Check regulatory path validation.

        Flags:
        - Surrogate endpoints in non-eligible indications
        - Designation stacking validation
        """
        flags: List[SanityFlag] = []

        # Group trials by ticker
        trials_by_ticker: Dict[str, List[TrialDetails]] = {}
        for trial in trials:
            if trial.ticker not in trials_by_ticker:
                trials_by_ticker[trial.ticker] = []
            trials_by_ticker[trial.ticker].append(trial)

        for ticker, ticker_trials in trials_by_ticker.items():
            # Check designation stacking (multiple designations for same asset)
            designations_per_asset: Dict[str, Set[str]] = {}

            for trial in ticker_trials:
                asset_key = f"{trial.indication}"
                if asset_key not in designations_per_asset:
                    designations_per_asset[asset_key] = set()

                if trial.has_orphan_designation:
                    designations_per_asset[asset_key].add("orphan")
                if trial.has_breakthrough_designation:
                    designations_per_asset[asset_key].add("breakthrough")
                if trial.has_fast_track:
                    designations_per_asset[asset_key].add("fast_track")

            # Validate stacked designations
            for asset_key, designations in designations_per_asset.items():
                if len(designations) >= 3:
                    # Triple designation - validate this is real
                    flags.append(SanityFlag(
                        severity=FlagSeverity.LOW,
                        category=CheckCategory.DOMAIN_EXPERT,
                        ticker=ticker,
                        check_name="designation_stack_validation",
                        message=f"Triple designation ({', '.join(designations)}) for {asset_key}",
                        details={
                            "indication": asset_key,
                            "designations": list(designations),
                        },
                        recommendation="Validate designation stack is legitimate vs. data duplication",
                    ))

        return flags

    def _check_partnerships(
        self,
        securities: List[SecurityContext],
        partnerships: List[PartnershipInfo],
    ) -> List[SanityFlag]:
        """
        Check partnership/platform coherence.

        Flags:
        - Major pharma partner but poor financial health
        - Platform concentration risk
        """
        flags: List[SanityFlag] = []

        sec_lookup = {s.ticker: s for s in securities}

        for partnership in partnerships:
            sec = sec_lookup.get(partnership.ticker)
            if not sec:
                continue

            # Check major pharma partnership with poor financials
            is_major = partnership.partner_name.lower() in self.MAJOR_PHARMA

            if is_major:
                if sec.financial_score is not None and sec.financial_score < Decimal("40"):
                    flags.append(SanityFlag(
                        severity=FlagSeverity.MEDIUM,
                        category=CheckCategory.DOMAIN_EXPERT,
                        ticker=partnership.ticker,
                        check_name="partnership_value_mismatch",
                        message=f"Major pharma partner ({partnership.partner_name}) but financial health D",
                        details={
                            "partner": partnership.partner_name,
                            "deal_value_mm": float(partnership.deal_value_mm) if partnership.deal_value_mm else None,
                            "financial_score": float(sec.financial_score),
                        },
                        recommendation="Partnership value not reflected OR terms unfavorable",
                    ))

        return flags

    def _check_stage_logic(
        self,
        securities: List[SecurityContext],
    ) -> List[SanityFlag]:
        """
        Check stage progression logic.

        Flags:
        - Pre-Phase 2 with high rank
        - Zero trials but ranked
        """
        flags: List[SanityFlag] = []

        for sec in securities:
            if sec.rank is None:
                continue

            # Pre-Phase 2 with high rank
            if sec.lead_phase in ("Preclinical", "Phase 1"):
                if sec.rank <= 20:
                    flags.append(SanityFlag(
                        severity=FlagSeverity.MEDIUM,
                        category=CheckCategory.DOMAIN_EXPERT,
                        ticker=sec.ticker,
                        check_name="early_stage_high_rank",
                        message=f"Pre-Phase 2 company ({sec.lead_phase}) ranked #{sec.rank}",
                        details={
                            "lead_phase": sec.lead_phase,
                            "rank": sec.rank,
                            "trial_count": sec.trial_count,
                        },
                        recommendation="Very early stage asset ranked highly - verify thesis",
                    ))

            # Pre-revenue with very low market cap should be flagged if highly ranked
            if sec.is_pre_revenue and sec.is_micro_cap:
                if sec.rank <= 10:
                    flags.append(SanityFlag(
                        severity=FlagSeverity.LOW,
                        category=CheckCategory.DOMAIN_EXPERT,
                        ticker=sec.ticker,
                        check_name="micro_cap_pre_revenue_top10",
                        message=f"Micro-cap pre-revenue company ranked #{sec.rank}",
                        details={
                            "market_cap_mm": float(sec.market_cap_mm) if sec.market_cap_mm else None,
                            "lead_phase": sec.lead_phase,
                            "rank": sec.rank,
                        },
                        recommendation="High risk profile for top ranking",
                    ))

        return flags

    def _is_oncology_indication(self, indication: str) -> bool:
        """Check if indication is oncology-related."""
        if not indication:
            return False
        oncology_terms = [
            "cancer", "tumor", "carcinoma", "lymphoma", "leukemia",
            "melanoma", "sarcoma", "glioma", "myeloma", "oncology",
            "neoplasm", "malignant", "metastatic",
        ]
        indication_lower = indication.lower()
        return any(term in indication_lower for term in oncology_terms)

    def _is_rare_disease(self, indication: str) -> bool:
        """Check if indication is likely rare disease."""
        if not indication:
            return False
        rare_terms = [
            "orphan", "rare", "duchenne", "huntington", "cystic fibrosis",
            "hemophilia", "sma", "spinal muscular", "als", "amyotrophic",
            "fabry", "gaucher", "pompe", "lysosomal", "mitochondrial",
        ]
        indication_lower = indication.lower()
        return any(term in indication_lower for term in rare_terms)

    def _calculate_metrics(
        self,
        flags: List[SanityFlag],
        securities: List[SecurityContext],
    ) -> Dict[str, Any]:
        """Calculate summary metrics."""
        check_types = [
            "underpowered_trial",
            "regulatory_path_risk",
            "crowded_market_risk",
            "partnership_value_mismatch",
            "early_stage_high_rank",
        ]

        by_check = {
            check: sum(1 for f in flags if check in f.check_name)
            for check in check_types
        }

        return {
            "total_flags": len(flags),
            "by_check": by_check,
            "by_severity": {
                "critical": sum(1 for f in flags if f.severity == FlagSeverity.CRITICAL),
                "high": sum(1 for f in flags if f.severity == FlagSeverity.HIGH),
                "medium": sum(1 for f in flags if f.severity == FlagSeverity.MEDIUM),
                "low": sum(1 for f in flags if f.severity == FlagSeverity.LOW),
            },
        }


def validate_trial_enrollment(
    phase: str,
    indication: str,
    enrollment: int,
) -> Tuple[bool, Optional[str]]:
    """
    Validate trial enrollment is sufficient.

    Args:
        phase: Trial phase
        indication: Disease indication
        enrollment: Number of patients

    Returns:
        (is_valid, reason_if_invalid)
    """
    checker = DomainExpertChecker()

    is_oncology = checker._is_oncology_indication(indication)
    is_rare = checker._is_rare_disease(indication)

    if phase == "Phase 3":
        if is_oncology:
            min_enrollment = DomainExpertChecker.MIN_ENROLLMENT["oncology_phase3"]
        elif is_rare:
            min_enrollment = DomainExpertChecker.MIN_ENROLLMENT["rare_disease_phase3"]
        else:
            min_enrollment = DomainExpertChecker.MIN_ENROLLMENT["default_phase3"]

        if enrollment < min_enrollment:
            return False, f"Enrollment {enrollment} below minimum {min_enrollment} for {phase}"

    elif phase == "Phase 2":
        if is_rare:
            min_enrollment = DomainExpertChecker.MIN_ENROLLMENT["rare_disease_phase2"]
        else:
            min_enrollment = DomainExpertChecker.MIN_ENROLLMENT["default_phase2"]

        if enrollment < min_enrollment:
            return False, f"Enrollment {enrollment} below minimum {min_enrollment} for {phase}"

    return True, None
