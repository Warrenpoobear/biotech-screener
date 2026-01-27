#!/usr/bin/env python3
"""
competitive_intensity_engine.py

Competitive Intensity Scoring Engine for Biotech Screener

Analyzes competitive landscape to assess:
- Indication crowding (number of competing programs)
- First-in-class vs me-too positioning
- Competitor stage advancement
- Approved drug competition

Key Insight: First-in-class programs in less crowded indications have
higher probability of commercial success, while me-too drugs in crowded
markets face pricing pressure and market share challenges.

Author: Wake Robin Capital Management
Version: 1.0.0
"""

import hashlib
import json
import re
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import date
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum


__version__ = "1.0.0"
__author__ = "Wake Robin Capital Management"


class CrowdingLevel(Enum):
    """Indication crowding classification."""
    UNCROWDED = "uncrowded"           # <5 competing programs
    MODERATE = "moderate"              # 5-15 competing programs
    CROWDED = "crowded"                # 16-30 competing programs
    HIGHLY_CROWDED = "highly_crowded"  # >30 competing programs


class CompetitivePosition(Enum):
    """Competitive positioning classification."""
    FIRST_IN_CLASS = "first_in_class"       # Novel mechanism, no competitors
    FAST_FOLLOWER = "fast_follower"         # Early in class, few competitors
    BEST_IN_CLASS = "best_in_class"         # Differentiated in crowded space
    ME_TOO = "me_too"                       # Undifferentiated in crowded space


@dataclass
class CompetitorProgram:
    """A competing program in the same indication."""
    ticker: str
    nct_id: str
    phase: str
    indication: str
    mechanism: str
    is_approved: bool = False


@dataclass
class IndicationLandscape:
    """Competitive landscape for an indication."""
    indication: str
    total_programs: int
    programs_by_phase: Dict[str, int]
    programs_by_ticker: Dict[str, int]
    has_approved_drug: bool
    most_advanced_phase: str
    unique_sponsors: int


class CompetitiveIntensityEngine:
    """
    Competitive intensity scoring for biotech investment analysis.

    Evaluates competitive pressure based on:
    - Number of competing programs in same indication
    - Development stage of competitors
    - Presence of approved drugs
    - First-in-class positioning

    Usage:
        engine = CompetitiveIntensityEngine()
        engine.build_landscape(trial_records, as_of_date)
        result = engine.score_ticker("ACME", as_of_date)
    """

    VERSION = "1.0.0"

    # Phase ordering for advancement comparison
    PHASE_ORDER = {
        "preclinical": 0,
        "phase_1": 1,
        "phase_1_2": 2,
        "phase_2": 3,
        "phase_2_3": 4,
        "phase_3": 5,
        "nda_bla": 6,
        "approved": 7,
    }

    # Crowding thresholds
    CROWDING_THRESHOLDS = {
        CrowdingLevel.UNCROWDED: (0, 5),
        CrowdingLevel.MODERATE: (5, 15),
        CrowdingLevel.CROWDED: (15, 30),
        CrowdingLevel.HIGHLY_CROWDED: (30, 9999),
    }

    # Score adjustments by crowding level (higher crowding = higher competitive pressure)
    CROWDING_SCORE_ADJUSTMENTS = {
        CrowdingLevel.UNCROWDED: Decimal("0"),        # Baseline - low competition
        CrowdingLevel.MODERATE: Decimal("15"),        # Some competition
        CrowdingLevel.CROWDED: Decimal("30"),         # Significant competition
        CrowdingLevel.HIGHLY_CROWDED: Decimal("45"),  # Intense competition
    }

    # Bonus/penalty for competitive position
    POSITION_ADJUSTMENTS = {
        CompetitivePosition.FIRST_IN_CLASS: Decimal("-20"),   # Advantage
        CompetitivePosition.FAST_FOLLOWER: Decimal("-10"),    # Some advantage
        CompetitivePosition.BEST_IN_CLASS: Decimal("0"),      # Neutral
        CompetitivePosition.ME_TOO: Decimal("15"),            # Disadvantage
    }

    # Approved drug penalty (existing competition)
    APPROVED_DRUG_PENALTY = Decimal("10")

    # Phase 3 competitor penalty
    PHASE_3_COMPETITOR_PENALTY = Decimal("5")

    # Indication category mappings (for grouping similar conditions)
    INDICATION_CATEGORIES = {
        "oncology": [
            "cancer", "tumor", "carcinoma", "leukemia", "lymphoma", "melanoma",
            "sarcoma", "myeloma", "glioma", "neuroblastoma", "nsclc", "sclc",
            "breast cancer", "lung cancer", "colorectal", "pancreatic", "ovarian",
            "prostate cancer", "renal cell", "hepatocellular", "gastric"
        ],
        "immunology": [
            "rheumatoid arthritis", "psoriasis", "lupus", "crohn", "colitis",
            "multiple sclerosis", "autoimmune", "inflammatory", "atopic dermatitis",
            "ankylosing spondylitis", "psoriatic arthritis"
        ],
        "neurology": [
            "alzheimer", "parkinson", "epilepsy", "migraine", "depression",
            "schizophrenia", "bipolar", "anxiety", "huntington", "als",
            "multiple sclerosis", "neuropathy", "stroke"
        ],
        "rare_disease": [
            "orphan", "rare", "duchenne", "sma", "cystic fibrosis", "hemophilia",
            "fabry", "gaucher", "pompe", "hunter", "hurler", "batten",
            "sickle cell", "thalassemia", "pku"
        ],
        "infectious": [
            "hiv", "hepatitis", "covid", "influenza", "rsv", "bacterial",
            "viral", "fungal", "tuberculosis", "malaria", "antibiotic"
        ],
        "cardiovascular": [
            "heart failure", "hypertension", "atrial fibrillation", "cad",
            "myocardial", "thrombosis", "stroke", "atherosclerosis", "pah"
        ],
        "metabolic": [
            "diabetes", "obesity", "nash", "fatty liver", "hyperlipidemia",
            "gout", "metabolic syndrome"
        ],
        "ophthalmology": [
            "macular degeneration", "diabetic retinopathy", "glaucoma",
            "uveitis", "dry eye", "retinal"
        ],
    }

    # Mechanism keywords for MOA grouping
    MECHANISM_KEYWORDS = {
        "checkpoint_inhibitor": ["pd-1", "pd-l1", "ctla-4", "lag-3", "tim-3", "tigit"],
        "car_t": ["car-t", "car t", "chimeric antigen"],
        "bispecific": ["bispecific", "bi-specific"],
        "adc": ["antibody-drug conjugate", "adc"],
        "kinase_inhibitor": ["kinase inhibitor", "tyrosine kinase", "jak", "btk", "pi3k"],
        "gene_therapy": ["gene therapy", "aav", "lentiviral", "crispr", "gene editing"],
        "rna_therapeutic": ["mrna", "sirna", "antisense", "aso", "rnai"],
        "antibody": ["monoclonal antibody", "mab", "antibody"],
        "cell_therapy": ["cell therapy", "stem cell", "ipsc"],
        "small_molecule": ["small molecule"],
        "vaccine": ["vaccine", "immunization"],
        "protein": ["fusion protein", "enzyme replacement"],
    }

    def __init__(self):
        """Initialize the competitive intensity engine."""
        self.indication_landscapes: Dict[str, IndicationLandscape] = {}
        self.ticker_programs: Dict[str, List[Dict]] = defaultdict(list)
        self.audit_trail: List[Dict[str, Any]] = []
        self._landscape_built = False

    def build_landscape(
        self,
        trial_records: List[Dict[str, Any]],
        as_of_date: date,
    ) -> Dict[str, Any]:
        """
        Build competitive landscape from trial records.

        Args:
            trial_records: List of clinical trial records
            as_of_date: Point-in-time date for filtering

        Returns:
            Dict with landscape statistics
        """
        self.indication_landscapes = {}
        self.ticker_programs = defaultdict(list)

        # Group trials by normalized indication
        indication_programs: Dict[str, List[Dict]] = defaultdict(list)

        for trial in trial_records:
            ticker = (
                trial.get("lead_sponsor_ticker") or
                trial.get("ticker") or ""
            ).upper()

            if not ticker:
                continue

            # Extract and normalize indication
            conditions = trial.get("conditions", [])
            if isinstance(conditions, str):
                conditions = [conditions]

            indication = self._normalize_indication(conditions)
            if not indication:
                continue

            # Extract phase
            phase = self._normalize_phase(trial.get("phase", ""))

            # Extract mechanism
            interventions = trial.get("interventions", [])
            mechanism = self._extract_mechanism(interventions)

            # Check if approved (simplified - look for "approved" status or commercial phase)
            status = trial.get("overall_status", "").lower()
            is_approved = status in ["approved", "marketed"] or phase == "approved"

            program = {
                "ticker": ticker,
                "nct_id": trial.get("nct_id", ""),
                "phase": phase,
                "indication": indication,
                "mechanism": mechanism,
                "is_approved": is_approved,
                "conditions_raw": conditions,
            }

            indication_programs[indication].append(program)
            self.ticker_programs[ticker].append(program)

        # Build landscape for each indication
        for indication, programs in indication_programs.items():
            # Count by phase
            phase_counts = Counter(p["phase"] for p in programs)

            # Count by ticker
            ticker_counts = Counter(p["ticker"] for p in programs)

            # Check for approved drugs
            has_approved = any(p["is_approved"] for p in programs)

            # Find most advanced phase
            most_advanced = "preclinical"
            for phase in ["approved", "nda_bla", "phase_3", "phase_2_3", "phase_2", "phase_1_2", "phase_1"]:
                if phase_counts.get(phase, 0) > 0:
                    most_advanced = phase
                    break

            self.indication_landscapes[indication] = IndicationLandscape(
                indication=indication,
                total_programs=len(programs),
                programs_by_phase=dict(phase_counts),
                programs_by_ticker=dict(ticker_counts),
                has_approved_drug=has_approved,
                most_advanced_phase=most_advanced,
                unique_sponsors=len(ticker_counts),
            )

        self._landscape_built = True

        return {
            "indications_mapped": len(self.indication_landscapes),
            "tickers_mapped": len(self.ticker_programs),
            "total_programs": sum(l.total_programs for l in self.indication_landscapes.values()),
            "top_indications": sorted(
                [(k, v.total_programs) for k, v in self.indication_landscapes.items()],
                key=lambda x: -x[1]
            )[:10],
        }

    def score_ticker(
        self,
        ticker: str,
        as_of_date: date,
    ) -> Dict[str, Any]:
        """
        Calculate competitive intensity score for a ticker.

        Args:
            ticker: Stock ticker
            as_of_date: Point-in-time date

        Returns:
            Dict containing:
            - competitive_intensity_score: 0-100 (higher = more competitive pressure)
            - crowding_level: Indication crowding classification
            - competitive_position: First-in-class vs me-too
            - competitor_count: Number of competing programs
            - phase_3_competitors: Count of Phase 3+ competitors
            - has_approved_competition: Whether approved drugs exist
        """
        ticker = ticker.upper()

        if not self._landscape_built:
            return self._no_data_result(ticker, as_of_date, "landscape_not_built")

        programs = self.ticker_programs.get(ticker, [])
        if not programs:
            return self._no_data_result(ticker, as_of_date, "no_programs_found")

        # Aggregate across all indications the ticker is in
        total_competitors = 0
        phase_3_plus_competitors = 0
        has_approved_competition = False
        indication_scores = []
        primary_indication = None
        primary_indication_crowding = CrowdingLevel.UNCROWDED

        # Find primary indication (most advanced program)
        programs_sorted = sorted(
            programs,
            key=lambda p: -self.PHASE_ORDER.get(p["phase"], 0)
        )

        if programs_sorted:
            primary_indication = programs_sorted[0]["indication"]

        seen_indications = set()
        for program in programs:
            indication = program["indication"]
            if indication in seen_indications:
                continue
            seen_indications.add(indication)

            landscape = self.indication_landscapes.get(indication)
            if not landscape:
                continue

            # Count competitors (excluding own programs)
            own_programs = landscape.programs_by_ticker.get(ticker, 0)
            competitors = landscape.total_programs - own_programs
            total_competitors += competitors

            # Count Phase 3+ competitors
            p3_plus = (
                landscape.programs_by_phase.get("phase_3", 0) +
                landscape.programs_by_phase.get("nda_bla", 0) +
                landscape.programs_by_phase.get("approved", 0)
            )
            # Subtract own Phase 3+ programs
            own_p3_plus = sum(
                1 for p in programs
                if p["indication"] == indication and p["phase"] in ["phase_3", "nda_bla", "approved"]
            )
            phase_3_plus_competitors += max(0, p3_plus - own_p3_plus)

            if landscape.has_approved_drug:
                has_approved_competition = True

            # Track crowding for primary indication
            if indication == primary_indication:
                primary_indication_crowding = self._classify_crowding(competitors)

        # Determine competitive position
        competitive_position = self._assess_position(
            ticker, programs, primary_indication, total_competitors
        )

        # Calculate score components
        base_score = Decimal("30")  # Baseline competitive pressure

        # Crowding adjustment
        crowding_adj = self.CROWDING_SCORE_ADJUSTMENTS[primary_indication_crowding]

        # Position adjustment
        position_adj = self.POSITION_ADJUSTMENTS[competitive_position]

        # Approved drug penalty
        approved_penalty = self.APPROVED_DRUG_PENALTY if has_approved_competition else Decimal("0")

        # Phase 3 competitor penalty (capped)
        p3_penalty = min(
            Decimal(str(phase_3_plus_competitors)) * self.PHASE_3_COMPETITOR_PENALTY,
            Decimal("20")
        )

        # Calculate final score
        intensity_score = (
            base_score +
            crowding_adj +
            position_adj +
            approved_penalty +
            p3_penalty
        )

        # Clamp to 0-100
        intensity_score = max(Decimal("0"), min(Decimal("100"), intensity_score))
        intensity_score = intensity_score.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)

        # Build audit entry
        audit_entry = {
            "ticker": ticker,
            "as_of_date": as_of_date.isoformat(),
            "primary_indication": primary_indication,
            "indications_count": len(seen_indications),
            "total_competitors": total_competitors,
            "phase_3_plus_competitors": phase_3_plus_competitors,
            "score_components": {
                "base_score": str(base_score),
                "crowding_adjustment": str(crowding_adj),
                "position_adjustment": str(position_adj),
                "approved_penalty": str(approved_penalty),
                "phase_3_penalty": str(p3_penalty),
                "final_score": str(intensity_score),
            },
        }
        self.audit_trail.append(audit_entry)

        return {
            "ticker": ticker,
            "competitive_intensity_score": intensity_score,
            "crowding_level": primary_indication_crowding.value,
            "competitive_position": competitive_position.value,
            "primary_indication": primary_indication,
            "competitor_count": total_competitors,
            "phase_3_competitors": phase_3_plus_competitors,
            "has_approved_competition": has_approved_competition,
            "indications_analyzed": len(seen_indications),
            "intensity_rating": self._get_intensity_rating(intensity_score),
            "audit_entry": audit_entry,
        }

    def score_universe(
        self,
        universe: List[Dict[str, Any]],
        trial_records: List[Dict[str, Any]],
        as_of_date: date,
    ) -> Dict[str, Any]:
        """
        Score competitive intensity for entire universe.

        Args:
            universe: List of dicts with 'ticker'
            trial_records: Clinical trial records
            as_of_date: Point-in-time date

        Returns:
            Dict with scores, diagnostics, and provenance
        """
        # Build landscape first
        landscape_stats = self.build_landscape(trial_records, as_of_date)

        scores = []
        intensity_distribution: Dict[str, int] = {
            "low": 0,      # 0-30
            "moderate": 0, # 31-50
            "high": 0,     # 51-70
            "intense": 0,  # 71-100
        }
        crowding_distribution: Dict[str, int] = {}
        position_distribution: Dict[str, int] = {}

        for company in universe:
            ticker = company.get("ticker", "").upper()
            if not ticker:
                continue

            result = self.score_ticker(ticker, as_of_date)

            scores.append({
                "ticker": ticker,
                "competitive_intensity_score": result["competitive_intensity_score"],
                "crowding_level": result["crowding_level"],
                "competitive_position": result["competitive_position"],
                "competitor_count": result["competitor_count"],
                "phase_3_competitors": result["phase_3_competitors"],
                "has_approved_competition": result["has_approved_competition"],
                "intensity_rating": result["intensity_rating"],
            })

            # Track distributions
            rating = result["intensity_rating"]
            intensity_distribution[rating] = intensity_distribution.get(rating, 0) + 1

            crowding = result["crowding_level"]
            crowding_distribution[crowding] = crowding_distribution.get(crowding, 0) + 1

            position = result["competitive_position"]
            position_distribution[position] = position_distribution.get(position, 0) + 1

        # Content hash for provenance
        scores_json = json.dumps(
            [{"t": s["ticker"], "s": str(s["competitive_intensity_score"])} for s in scores],
            sort_keys=True
        )
        content_hash = hashlib.sha256(scores_json.encode()).hexdigest()[:16]

        return {
            "as_of_date": as_of_date.isoformat(),
            "scores": scores,
            "landscape_stats": landscape_stats,
            "diagnostic_counts": {
                "total_scored": len(scores),
                "intensity_distribution": intensity_distribution,
                "crowding_distribution": crowding_distribution,
                "position_distribution": position_distribution,
                "with_approved_competition": sum(1 for s in scores if s["has_approved_competition"]),
                "avg_competitor_count": sum(s["competitor_count"] for s in scores) / max(1, len(scores)),
            },
            "provenance": {
                "module": "competitive_intensity_engine",
                "module_version": self.VERSION,
                "content_hash": content_hash,
                "pit_cutoff": as_of_date.isoformat(),
            },
        }

    def _normalize_indication(self, conditions: List[str]) -> Optional[str]:
        """Normalize conditions to indication category."""
        if not conditions:
            return None

        conditions_text = " ".join(str(c).lower() for c in conditions)

        # Check each category
        for category, keywords in self.INDICATION_CATEGORIES.items():
            if any(kw in conditions_text for kw in keywords):
                return category

        # Return first condition as fallback
        return conditions[0].lower()[:50] if conditions else None

    def _normalize_phase(self, phase: str) -> str:
        """Normalize phase string."""
        if not phase:
            return "phase_2"

        phase_lower = phase.lower().strip()

        phase_map = {
            "early phase 1": "phase_1",
            "phase 1": "phase_1",
            "phase1": "phase_1",
            "phase i": "phase_1",
            "phase 1/phase 2": "phase_1_2",
            "phase 1/2": "phase_1_2",
            "phase 2": "phase_2",
            "phase2": "phase_2",
            "phase ii": "phase_2",
            "phase 2/phase 3": "phase_2_3",
            "phase 2/3": "phase_2_3",
            "phase 3": "phase_3",
            "phase3": "phase_3",
            "phase iii": "phase_3",
            "phase 4": "approved",
            "phase iv": "approved",
            "not applicable": "approved",
            "n/a": "approved",
        }

        return phase_map.get(phase_lower, "phase_2")

    def _extract_mechanism(self, interventions: Any) -> str:
        """Extract mechanism of action from interventions."""
        if not interventions:
            return "unknown"

        if isinstance(interventions, str):
            interventions = [interventions]

        intervention_text = " ".join(str(i).lower() for i in interventions)

        for mechanism, keywords in self.MECHANISM_KEYWORDS.items():
            if any(kw in intervention_text for kw in keywords):
                return mechanism

        return "other"

    def _classify_crowding(self, competitor_count: int) -> CrowdingLevel:
        """Classify crowding level based on competitor count."""
        for level, (min_val, max_val) in self.CROWDING_THRESHOLDS.items():
            if min_val <= competitor_count < max_val:
                return level
        return CrowdingLevel.HIGHLY_CROWDED

    def _assess_position(
        self,
        ticker: str,
        programs: List[Dict],
        primary_indication: Optional[str],
        total_competitors: int,
    ) -> CompetitivePosition:
        """Assess competitive position."""
        if total_competitors < 3:
            return CompetitivePosition.FIRST_IN_CLASS
        elif total_competitors < 10:
            return CompetitivePosition.FAST_FOLLOWER
        elif total_competitors < 25:
            return CompetitivePosition.BEST_IN_CLASS
        else:
            return CompetitivePosition.ME_TOO

    def _get_intensity_rating(self, score: Decimal) -> str:
        """Get human-readable intensity rating."""
        if score <= 30:
            return "low"
        elif score <= 50:
            return "moderate"
        elif score <= 70:
            return "high"
        else:
            return "intense"

    def _no_data_result(
        self,
        ticker: str,
        as_of_date: date,
        reason: str
    ) -> Dict[str, Any]:
        """Return result when no data available."""
        return {
            "ticker": ticker,
            "competitive_intensity_score": Decimal("50"),  # Neutral
            "crowding_level": "unknown",
            "competitive_position": "unknown",
            "primary_indication": None,
            "competitor_count": 0,
            "phase_3_competitors": 0,
            "has_approved_competition": False,
            "indications_analyzed": 0,
            "intensity_rating": "unknown",
            "audit_entry": {
                "ticker": ticker,
                "as_of_date": as_of_date.isoformat(),
                "note": reason,
            },
        }

    def get_indication_landscape(self, indication: str) -> Optional[IndicationLandscape]:
        """Get landscape for a specific indication."""
        return self.indication_landscapes.get(indication)

    def get_top_competitors(
        self,
        ticker: str,
        indication: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get top competitors in an indication."""
        landscape = self.indication_landscapes.get(indication)
        if not landscape:
            return []

        competitors = []
        for comp_ticker, count in landscape.programs_by_ticker.items():
            if comp_ticker != ticker.upper():
                competitors.append({
                    "ticker": comp_ticker,
                    "program_count": count,
                })

        return sorted(competitors, key=lambda x: -x["program_count"])[:limit]

    def get_audit_trail(self) -> List[Dict[str, Any]]:
        """Return the audit trail."""
        return self.audit_trail.copy()

    def clear_audit_trail(self) -> None:
        """Clear the audit trail."""
        self.audit_trail = []


if __name__ == "__main__":
    # Demonstration
    print("=" * 70)
    print("COMPETITIVE INTENSITY ENGINE v1.0.0 - DEMONSTRATION")
    print("=" * 70)

    engine = CompetitiveIntensityEngine()

    # Create sample trial records
    sample_trials = [
        {"lead_sponsor_ticker": "ACME", "nct_id": "NCT001", "phase": "Phase 2",
         "conditions": ["Breast Cancer"], "interventions": ["monoclonal antibody"]},
        {"lead_sponsor_ticker": "ACME", "nct_id": "NCT002", "phase": "Phase 3",
         "conditions": ["Lung Cancer"], "interventions": ["kinase inhibitor"]},
        {"lead_sponsor_ticker": "COMP1", "nct_id": "NCT003", "phase": "Phase 3",
         "conditions": ["Breast Cancer"], "interventions": ["antibody"]},
        {"lead_sponsor_ticker": "COMP2", "nct_id": "NCT004", "phase": "Phase 2",
         "conditions": ["Breast Cancer"], "interventions": ["adc"]},
        {"lead_sponsor_ticker": "COMP3", "nct_id": "NCT005", "phase": "Phase 1",
         "conditions": ["Breast Cancer"], "interventions": ["car-t"]},
    ]

    as_of = date(2026, 1, 26)

    # Build landscape
    stats = engine.build_landscape(sample_trials, as_of)
    print(f"\nLandscape built: {stats['indications_mapped']} indications, "
          f"{stats['tickers_mapped']} tickers")

    # Score a ticker
    result = engine.score_ticker("ACME", as_of)
    print(f"\nACME Competitive Analysis:")
    print(f"  Intensity Score: {result['competitive_intensity_score']}")
    print(f"  Crowding Level: {result['crowding_level']}")
    print(f"  Position: {result['competitive_position']}")
    print(f"  Competitors: {result['competitor_count']}")
    print(f"  Phase 3+ Competitors: {result['phase_3_competitors']}")
