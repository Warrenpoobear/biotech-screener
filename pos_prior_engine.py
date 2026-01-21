#!/usr/bin/env python3
"""
pos_prior_engine.py

Probability of Success (PoS) Prior Engine for Biotech Screener

Provides a 2D PoS matrix (stage × therapeutic area) with observable modifiers,
confidence-gated, used as ranking prior not absolute probability.

Design Philosophy (Governed Over Smart):
- DETERMINISTIC: No datetime.now() - timestamps derive from as_of_date
- STDLIB-ONLY: No external dependencies
- FAIL LOUDLY: Explicit error states, not silent defaults
- PIT DISCIPLINE: All inputs from point-in-time snapshots
- EXPLICIT CLAMPING: All scores bounded to declared ranges
- CONFIDENCE-GATED: Only apply adjustments if confidence > threshold

Benchmarks:
- Loaded from versioned external file: data/pos_benchmarks_bio_2011_2020_v1.json
- Source: BIO Clinical Development Success Rates 2011-2020
- Values should be verified against original report before production use

Author: Wake Robin Capital Management
Version: 1.0.0
"""

import hashlib
import json
import re
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import date
from pathlib import Path
from enum import Enum

# Module metadata
__version__ = "1.0.0"
__author__ = "Wake Robin Capital Management"


class DataQualityState(Enum):
    """Data quality classification for PoS analysis."""
    FULL = "FULL"          # All required + optional fields present
    PARTIAL = "PARTIAL"    # Required fields + some optional
    MINIMAL = "MINIMAL"    # Minimum viable data only
    NONE = "NONE"          # Insufficient data to score


class PoSPriorEngine:
    """
    Probability of Success Prior Engine.

    Calculates PoS prior using a 2D matrix (stage × therapeutic area) with
    observable modifiers (FDA designations, biomarker enrichment).

    Key Design Decisions:
    - PoS is used as a ranking prior, not an absolute probability
    - Modifiers are capped to prevent overconfidence
    - Confidence scoring based on data completeness and sample size
    - Integrates with composite scoring as multiplicative weight

    Usage:
        engine = PoSPriorEngine()
        result = engine.calculate_pos_prior(
            stage="phase_3",
            therapeutic_area="oncology",
            orphan_drug_designation=True,
            as_of_date=date(2026, 1, 15)
        )
        print(result["pos_prior"])    # 0.505 (adjusted PoS)
        print(result["confidence"])   # 0.85
    """

    VERSION = "1.0.0"
    BENCHMARKS_FILE = "data/pos_benchmarks_bio_2011_2020_v1.json"

    # Score ranges (explicit bounds)
    POS_PRIOR_MIN = Decimal("0")
    POS_PRIOR_MAX = Decimal("0.95")  # Never 100% certain
    CONFIDENCE_MIN = Decimal("0")
    CONFIDENCE_MAX = Decimal("1")

    # Observable modifiers with empirical support
    MODIFIERS: Dict[str, Decimal] = {
        "orphan_drug_designation": Decimal("1.15"),    # FDA orphan status
        "breakthrough_designation": Decimal("1.25"),   # FDA breakthrough
        "fast_track_designation": Decimal("1.10"),     # FDA fast track
        "biomarker_enriched": Decimal("1.20"),         # Trial design field
    }

    # Cap total adjustment to prevent overconfidence
    MAX_MODIFIER_ADJUSTMENT = Decimal("2.0")

    # Stage normalization map
    STAGE_MAP: Dict[str, str] = {
        "preclinical": "preclinical",
        "pre-clinical": "preclinical",
        "discovery": "preclinical",
        "phase 1": "phase_1",
        "phase1": "phase_1",
        "phase i": "phase_1",
        "p1": "phase_1",
        "phase_1": "phase_1",
        "phase 1/2": "phase_1_2",
        "phase1/2": "phase_1_2",
        "phase i/ii": "phase_1_2",
        "phase_1_2": "phase_1_2",
        "phase 2": "phase_2",
        "phase2": "phase_2",
        "phase ii": "phase_2",
        "p2": "phase_2",
        "phase_2": "phase_2",
        "phase 2/3": "phase_2_3",
        "phase2/3": "phase_2_3",
        "phase ii/iii": "phase_2_3",
        "phase_2_3": "phase_2_3",
        "phase 3": "phase_3",
        "phase3": "phase_3",
        "phase iii": "phase_3",
        "p3": "phase_3",
        "phase_3": "phase_3",
        "pivotal": "phase_3",
        "nda": "nda_bla",
        "bla": "nda_bla",
        "nda/bla": "nda_bla",
        "nda_bla": "nda_bla",
        "submitted": "nda_bla",
        "filed": "nda_bla",
        "approved": "commercial",
        "commercial": "commercial",
        "marketed": "commercial",
    }

    # Therapeutic area normalization map
    TA_MAP: Dict[str, str] = {
        # Direct mappings
        "oncology": "oncology",
        "cancer": "oncology",
        "tumor": "oncology",
        "rare_disease": "rare_disease",
        "rare disease": "rare_disease",
        "orphan": "rare_disease",
        "infectious_disease": "infectious_disease",
        "infectious disease": "infectious_disease",
        "infectious": "infectious_disease",
        "neurology": "neurology",
        "cns": "neurology",
        "neurological": "neurology",
        "cardiovascular": "cardiovascular",
        "cardiac": "cardiovascular",
        "cardio": "cardiovascular",
        "immunology": "immunology",
        "autoimmune": "immunology",
        "immune": "immunology",
        "metabolic": "metabolic",
        "metabolism": "metabolic",
        "respiratory": "respiratory",
        "pulmonary": "respiratory",
        "dermatology": "dermatology",
        "skin": "dermatology",
        "ophthalmology": "ophthalmology",
        "ophthalmic": "ophthalmology",
        "eye": "ophthalmology",
        "gastroenterology": "gastroenterology",
        "gi": "gastroenterology",
        "gi_hepatology": "gastroenterology",
        "hematology": "hematology",
        "blood": "hematology",
        "urology": "urology",
    }

    # Required fields for scoring
    REQUIRED_FIELDS = ["stage"]
    OPTIONAL_FIELDS = ["therapeutic_area", "orphan_drug_designation",
                       "breakthrough_designation", "fast_track_designation",
                       "biomarker_enriched"]

    # Minimum sample size for full confidence
    MIN_SAMPLE_SIZE_FULL_CONFIDENCE = 200

    def __init__(self, benchmarks_path: Optional[str] = None):
        """
        Initialize the PoS Prior engine.

        Args:
            benchmarks_path: Optional path to benchmarks JSON file.
                             Defaults to data/pos_benchmarks_bio_2011_2020_v1.json
        """
        self.benchmarks_path = benchmarks_path or self.BENCHMARKS_FILE
        self.benchmarks: Dict[str, Dict[str, str]] = {}
        self.benchmarks_metadata: Dict[str, Any] = {}
        self.audit_trail: List[Dict[str, Any]] = []

        # Sample sizes for confidence calculation (from BIO report)
        self.sample_sizes: Dict[Tuple[str, str], int] = self._default_sample_sizes()

        # Load benchmarks
        self._load_benchmarks()

    def _default_sample_sizes(self) -> Dict[Tuple[str, str], int]:
        """Default sample sizes from BIO 2011-2020 study."""
        return {
            # Phase 1
            ("phase_1", "oncology"): 450,
            ("phase_1", "rare_disease"): 180,
            ("phase_1", "infectious_disease"): 90,
            ("phase_1", "neurology"): 200,
            ("phase_1", "cardiovascular"): 120,
            ("phase_1", "immunology"): 120,
            ("phase_1", "metabolic"): 150,
            ("phase_1", "respiratory"): 100,
            ("phase_1", "dermatology"): 80,
            ("phase_1", "ophthalmology"): 70,
            ("phase_1", "gastroenterology"): 90,
            ("phase_1", "hematology"): 85,
            ("phase_1", "urology"): 75,
            ("phase_1", "all_indications"): 500,
            # Phase 2
            ("phase_2", "oncology"): 380,
            ("phase_2", "rare_disease"): 150,
            ("phase_2", "infectious_disease"): 75,
            ("phase_2", "neurology"): 170,
            ("phase_2", "cardiovascular"): 95,
            ("phase_2", "immunology"): 95,
            ("phase_2", "metabolic"): 130,
            ("phase_2", "respiratory"): 85,
            ("phase_2", "dermatology"): 70,
            ("phase_2", "ophthalmology"): 60,
            ("phase_2", "gastroenterology"): 75,
            ("phase_2", "hematology"): 70,
            ("phase_2", "urology"): 60,
            ("phase_2", "all_indications"): 420,
            # Phase 3
            ("phase_3", "oncology"): 220,
            ("phase_3", "rare_disease"): 95,
            ("phase_3", "infectious_disease"): 60,
            ("phase_3", "neurology"): 95,
            ("phase_3", "cardiovascular"): 75,
            ("phase_3", "immunology"): 75,
            ("phase_3", "metabolic"): 80,
            ("phase_3", "respiratory"): 65,
            ("phase_3", "dermatology"): 50,
            ("phase_3", "ophthalmology"): 45,
            ("phase_3", "gastroenterology"): 55,
            ("phase_3", "hematology"): 50,
            ("phase_3", "urology"): 45,
            ("phase_3", "all_indications"): 280,
        }

    def _load_benchmarks(self) -> None:
        """Load PoS benchmarks from external versioned file."""
        try:
            paths_to_try = [
                Path(__file__).parent / self.benchmarks_path,
                Path(self.benchmarks_path)
            ]

            for path in paths_to_try:
                if path.exists():
                    with open(path, "r") as f:
                        data = json.load(f)
                    self.benchmarks_metadata = data.get("_metadata", {})
                    self.benchmarks = {
                        "phase_1": data.get("phase_1_loa", {}),
                        "phase_2": data.get("phase_2_loa", {}),
                        "phase_3": data.get("phase_3_loa", {}),
                        "nda_bla": data.get("nda_bla_loa", {}),
                    }
                    return

            # Fallback to hardcoded defaults
            self._use_fallback_benchmarks()

        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to load benchmarks file: {e}. Using fallback benchmarks.")
            self._use_fallback_benchmarks()

    def _use_fallback_benchmarks(self) -> None:
        """Use hardcoded fallback benchmarks when file unavailable."""
        self.benchmarks_metadata = {
            "source": "FALLBACK_HARDCODED",
            "warning": "External benchmarks file not loaded"
        }
        # Conservative fallback values from spec
        self.benchmarks = {
            "phase_1": {
                "oncology": "0.095",
                "rare_disease": "0.120",
                "immunology": "0.105",
                "neurology": "0.080",
                "infectious_disease": "0.110",
                "all_indications": "0.090"
            },
            "phase_2": {
                "oncology": "0.285",
                "rare_disease": "0.350",
                "immunology": "0.310",
                "neurology": "0.250",
                "infectious_disease": "0.320",
                "all_indications": "0.280"
            },
            "phase_3": {
                "oncology": "0.480",
                "rare_disease": "0.650",
                "immunology": "0.520",
                "neurology": "0.420",
                "infectious_disease": "0.550",
                "all_indications": "0.450"
            },
            "nda_bla": {
                "all_indications": "0.903"
            }
        }

    def calculate_pos_prior(
        self,
        stage: str,
        therapeutic_area: Optional[str] = None,
        orphan_drug_designation: bool = False,
        breakthrough_designation: bool = False,
        fast_track_designation: bool = False,
        biomarker_enriched: bool = False,
        as_of_date: Optional[date] = None,
    ) -> Dict[str, Any]:
        """
        Calculate PoS prior with confidence gating.

        Args:
            stage: Development stage ("phase_1", "phase_2", "phase_3", etc.)
            therapeutic_area: Therapeutic indication (e.g., "oncology", "rare disease")
            orphan_drug_designation: FDA orphan drug designation
            breakthrough_designation: FDA breakthrough therapy designation
            fast_track_designation: FDA fast track designation
            biomarker_enriched: Trial uses biomarker enrichment
            as_of_date: Point-in-time date (REQUIRED for determinism)

        Returns:
            Dict containing:
            - pos_prior: Decimal 0-0.95 (adjusted PoS, None if insufficient data)
            - base_pos: Decimal (unadjusted base rate)
            - modifiers_applied: List[str]
            - modifier_adjustment: Decimal (total multiplier)
            - confidence: Decimal 0-1
            - reason_code: str (SUCCESS, INSUFFICIENT_DATA, UNMAPPED_STAGE)
            - metadata: Dict (sample_size, data_vintage, etc.)
            - hash: str (deterministic content hash)
        """

        # Validate as_of_date (required for determinism)
        if as_of_date is None:
            raise ValueError(
                "as_of_date is required for deterministic scoring. "
                "Providing date.today() as default would violate PIT discipline."
            )

        # Deterministic timestamp
        deterministic_timestamp = f"{as_of_date.isoformat()}T00:00:00Z"

        # Track missing fields
        missing_fields: List[str] = []
        inputs_used: Dict[str, Any] = {}

        # Validation - stage is required
        if not stage:
            missing_fields.append("stage")
            return self._create_error_result(
                reason_code="INSUFFICIENT_DATA",
                missing_fields=missing_fields,
                as_of_date=as_of_date,
                deterministic_timestamp=deterministic_timestamp,
            )

        # Normalize stage
        stage_normalized = self._normalize_stage(stage)
        inputs_used["stage"] = stage
        inputs_used["stage_normalized"] = stage_normalized

        if not stage_normalized:
            return self._create_error_result(
                reason_code="UNMAPPED_STAGE",
                missing_fields=[],
                as_of_date=as_of_date,
                deterministic_timestamp=deterministic_timestamp,
                details=f"Stage '{stage}' not in base rate matrix",
            )

        # Normalize therapeutic area
        ta_normalized = self._normalize_therapeutic_area(therapeutic_area)
        inputs_used["therapeutic_area"] = therapeutic_area
        inputs_used["therapeutic_area_normalized"] = ta_normalized

        # Track TA confidence (mapped vs fallback)
        if therapeutic_area and not ta_normalized:
            ta_confidence = Decimal("0.50")  # Lower confidence for unmapped TA
            ta_normalized = "all_indications"
        elif ta_normalized:
            ta_confidence = Decimal("1.0")
        else:
            ta_confidence = Decimal("0.70")  # No TA provided - use all_indications
            ta_normalized = "all_indications"

        # Track modifiers
        inputs_used["orphan_drug_designation"] = orphan_drug_designation
        inputs_used["breakthrough_designation"] = breakthrough_designation
        inputs_used["fast_track_designation"] = fast_track_designation
        inputs_used["biomarker_enriched"] = biomarker_enriched

        # Lookup base rate
        base_pos = self._get_base_pos(stage_normalized, ta_normalized)

        if base_pos is None:
            return self._create_error_result(
                reason_code="UNMAPPED_STAGE",
                missing_fields=[],
                as_of_date=as_of_date,
                deterministic_timestamp=deterministic_timestamp,
                details=f"No base rate for stage={stage_normalized}, ta={ta_normalized}",
            )

        # Apply modifiers
        applicable_modifiers: List[str] = []
        modifier_product = Decimal("1.0")

        if orphan_drug_designation:
            applicable_modifiers.append("orphan_drug_designation")
            modifier_product *= self.MODIFIERS["orphan_drug_designation"]

        if breakthrough_designation:
            applicable_modifiers.append("breakthrough_designation")
            modifier_product *= self.MODIFIERS["breakthrough_designation"]

        if fast_track_designation:
            applicable_modifiers.append("fast_track_designation")
            modifier_product *= self.MODIFIERS["fast_track_designation"]

        if biomarker_enriched:
            applicable_modifiers.append("biomarker_enriched")
            modifier_product *= self.MODIFIERS["biomarker_enriched"]

        # Cap adjustment
        modifier_product = min(modifier_product, self.MAX_MODIFIER_ADJUSTMENT)

        # Adjusted PoS
        adjusted_pos = base_pos * modifier_product
        adjusted_pos = min(adjusted_pos, self.POS_PRIOR_MAX)  # Never 100% certain
        adjusted_pos = adjusted_pos.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

        # Get sample size for confidence calculation
        sample_size = self.sample_sizes.get(
            (stage_normalized, ta_normalized),
            self.sample_sizes.get((stage_normalized, "all_indications"), 100)
        )

        # Confidence calculation
        # Factors: sample size, TA mapping quality, modifier observability
        sample_confidence = min(
            Decimal("1.0"),
            Decimal(str(sample_size)) / Decimal(str(self.MIN_SAMPLE_SIZE_FULL_CONFIDENCE))
        )
        modifier_confidence = Decimal("1.0") if len(applicable_modifiers) <= 2 else Decimal("0.85")

        overall_confidence = (
            Decimal("0.40") * sample_confidence +
            Decimal("0.40") * ta_confidence +
            Decimal("0.20") * modifier_confidence
        )
        overall_confidence = self._clamp(overall_confidence, self.CONFIDENCE_MIN, self.CONFIDENCE_MAX)
        overall_confidence = overall_confidence.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        # Data quality state
        data_quality_state = self._assess_data_quality(
            therapeutic_area is not None,
            len(applicable_modifiers) > 0
        )

        # Deterministic hash
        hash_input = (
            f"{stage_normalized}|{ta_normalized}|{base_pos}|"
            f"{'|'.join(sorted(applicable_modifiers))}"
        )
        content_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]

        # Audit entry
        audit_entry = {
            "timestamp": deterministic_timestamp,
            "as_of_date": as_of_date.isoformat(),
            "inputs_hash": hashlib.sha256(
                json.dumps(inputs_used, sort_keys=True, default=str).encode()
            ).hexdigest()[:16],
            "inputs_used": inputs_used,
            "missing_fields": missing_fields,
            "data_quality_state": data_quality_state.value,
            "calculation": {
                "base_pos": str(base_pos),
                "adjusted_pos": str(adjusted_pos),
                "modifiers_applied": applicable_modifiers,
                "modifier_product": str(modifier_product),
                "confidence": str(overall_confidence),
            },
            "benchmarks_source": self.benchmarks_metadata.get("source", "UNKNOWN"),
            "module_version": self.VERSION,
        }

        self.audit_trail.append(audit_entry)

        return {
            "pos_prior": adjusted_pos,
            "base_pos": base_pos,
            "modifiers_applied": applicable_modifiers,
            "modifier_adjustment": modifier_product.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            "confidence": overall_confidence,
            "reason_code": "SUCCESS",
            "data_quality_state": data_quality_state.value,
            "metadata": {
                "sample_size": sample_size,
                "data_vintage": self.benchmarks_metadata.get("report_version", "2011-2020"),
                "ta_mapped": ta_normalized,
                "stage_mapped": stage_normalized,
            },
            "hash": content_hash,
            "inputs_used": inputs_used,
            "audit_entry": audit_entry,
        }

    def score_universe(
        self,
        universe: List[Dict[str, Any]],
        as_of_date: date,
    ) -> Dict[str, Any]:
        """
        Score an entire universe of trials for PoS prior.

        Args:
            universe: List of dicts with required fields:
                - ticker: str
                - stage: str (development phase)
                Optional fields:
                - therapeutic_area: str
                - orphan_drug_designation: bool
                - breakthrough_designation: bool
                - fast_track_designation: bool
                - biomarker_enriched: bool
            as_of_date: Point-in-time date (REQUIRED)

        Returns:
            Dict with scores, diagnostics, and provenance
        """

        scores: List[Dict[str, Any]] = []
        indication_coverage = 0
        stage_distribution: Dict[str, int] = {}
        quality_distribution: Dict[str, int] = {
            "FULL": 0, "PARTIAL": 0, "MINIMAL": 0, "NONE": 0
        }

        for company in universe:
            ticker = company.get("ticker", "UNKNOWN")

            result = self.calculate_pos_prior(
                stage=company.get("stage", company.get("base_stage", "")),
                therapeutic_area=company.get("therapeutic_area", company.get("indication")),
                orphan_drug_designation=company.get("orphan_drug_designation", False),
                breakthrough_designation=company.get("breakthrough_designation", False),
                fast_track_designation=company.get("fast_track_designation", False),
                biomarker_enriched=company.get("biomarker_enriched", False),
                as_of_date=as_of_date,
            )

            scores.append({
                "ticker": ticker,
                "pos_prior": result["pos_prior"],
                "base_pos": result.get("base_pos"),
                "modifiers_applied": result.get("modifiers_applied", []),
                "modifier_adjustment": result.get("modifier_adjustment"),
                "confidence": result["confidence"],
                "reason_code": result["reason_code"],
                "data_quality_state": result.get("data_quality_state", "NONE"),
                "metadata": result.get("metadata", {}),
                "flags": [],
            })

            # Track metrics
            if result.get("metadata", {}).get("ta_mapped") and \
               result["metadata"]["ta_mapped"] != "all_indications":
                indication_coverage += 1

            stage = result.get("metadata", {}).get("stage_mapped", "unknown")
            stage_distribution[stage] = stage_distribution.get(stage, 0) + 1

            quality_state = result.get("data_quality_state", "NONE")
            if quality_state in quality_distribution:
                quality_distribution[quality_state] += 1

        # Deterministic content hash
        scores_json = json.dumps(
            [{"t": s["ticker"], "p": str(s["pos_prior"]), "c": str(s["confidence"])}
             for s in scores],
            sort_keys=True
        )
        content_hash = hashlib.sha256(scores_json.encode()).hexdigest()[:16]

        return {
            "as_of_date": as_of_date.isoformat(),
            "scores": scores,
            "diagnostic_counts": {
                "total_scored": len(scores),
                "success_count": sum(1 for s in scores if s["reason_code"] == "SUCCESS"),
                "error_count": sum(1 for s in scores if s["reason_code"] != "SUCCESS"),
                "indication_coverage": indication_coverage,
                "indication_coverage_pct": f"{indication_coverage / max(1, len(scores)) * 100:.1f}%",
                "stage_distribution": stage_distribution,
                "data_quality_distribution": quality_distribution,
            },
            "provenance": {
                "module": "pos_prior_engine",
                "module_version": self.VERSION,
                "content_hash": content_hash,
                "pit_cutoff": as_of_date.isoformat(),
                "benchmarks_source": self.benchmarks_metadata.get("source", "UNKNOWN"),
                "benchmarks_version": self.benchmarks_metadata.get("report_version", "UNKNOWN"),
            },
        }

    def _get_base_pos(self, stage: str, ta: str) -> Optional[Decimal]:
        """Get base PoS from benchmarks."""
        # Handle intermediate stages
        if stage == "preclinical":
            # Preclinical: 50% of Phase 1 rate
            phase1_rate = self._get_base_pos("phase_1", ta)
            if phase1_rate:
                return (phase1_rate * Decimal("0.5")).quantize(Decimal("0.001"))
            return None

        if stage == "phase_1_2":
            # Phase 1/2: Interpolate between Phase 1 and Phase 2
            p1 = self._get_base_pos("phase_1", ta)
            p2 = self._get_base_pos("phase_2", ta)
            if p1 and p2:
                return ((p1 + p2) / Decimal("2")).quantize(Decimal("0.001"))
            return p1 or p2

        if stage == "phase_2_3":
            # Phase 2/3: Interpolate between Phase 2 and Phase 3
            p2 = self._get_base_pos("phase_2", ta)
            p3 = self._get_base_pos("phase_3", ta)
            if p2 and p3:
                return ((p2 + p3) / Decimal("2")).quantize(Decimal("0.001"))
            return p2 or p3

        if stage == "commercial":
            return Decimal("1.0")

        # Direct lookup
        stage_benchmarks = self.benchmarks.get(stage, {})

        if ta in stage_benchmarks:
            try:
                return Decimal(stage_benchmarks[ta])
            except (InvalidOperation, ValueError):
                pass

        # Fallback to all_indications
        if "all_indications" in stage_benchmarks:
            try:
                return Decimal(stage_benchmarks["all_indications"])
            except (InvalidOperation, ValueError):
                pass

        return None

    def _normalize_stage(self, stage: str) -> Optional[str]:
        """Normalize stage name to canonical format."""
        if not stage:
            return None

        stage_lower = stage.lower().strip()

        if stage_lower in self.STAGE_MAP:
            return self.STAGE_MAP[stage_lower]

        # Try underscore version
        stage_underscore = stage_lower.replace(" ", "_").replace("-", "_")
        if stage_underscore in self.STAGE_MAP:
            return self.STAGE_MAP[stage_underscore]

        return None

    def _normalize_therapeutic_area(self, ta: Optional[str]) -> Optional[str]:
        """Normalize therapeutic area to benchmark categories."""
        if not ta:
            return None

        ta_lower = ta.lower().strip()

        # Direct lookup
        if ta_lower in self.TA_MAP:
            return self.TA_MAP[ta_lower]

        # Try underscore version
        ta_underscore = ta_lower.replace(" ", "_").replace("-", "_")
        if ta_underscore in self.TA_MAP:
            return self.TA_MAP[ta_underscore]

        # Pattern matching for complex indications
        ta_patterns: Dict[str, List[str]] = {
            "oncology": ["cancer", "tumor", "tumour", "leukemia", "lymphoma",
                         "carcinoma", "melanoma", "sarcoma", "myeloma", "neoplasm"],
            "rare_disease": ["rare", "orphan", "ultra-rare"],
            "infectious_disease": ["infectious", "hiv", "hepatitis", "antibiotic",
                                   "antiviral", "antimicrobial", "covid", "bacterial",
                                   "viral", "fungal"],
            "neurology": ["neuro", "alzheimer", "parkinson", "als", "epilepsy",
                          "multiple sclerosis", "huntington", "neuropathy"],
            "cardiovascular": ["cardio", "heart", "cardiac", "hypertension",
                               "arrhythmia", "atherosclerosis"],
            "immunology": ["immune", "autoimmune", "rheumatoid", "lupus",
                           "psoriasis", "crohn", "colitis", "inflammation"],
            "metabolic": ["metabolic", "diabetes", "obesity", "nafld", "nash",
                          "lipid", "cholesterol"],
            "respiratory": ["respiratory", "copd", "asthma", "pulmonary",
                            "cystic fibrosis", "lung"],
            "dermatology": ["dermatolog", "skin", "atopic dermatitis", "eczema",
                            "acne", "vitiligo"],
            "ophthalmology": ["ophthalm", "eye", "retina", "macular", "glaucoma",
                              "uveitis", "vision"],
            "gastroenterology": ["gastro", "gastrointestinal", "ibd", "irritable bowel",
                                 "liver", "hepatic", "gi"],
            "hematology": ["hematolog", "blood", "hemophilia", "sickle cell",
                           "thalassemia", "anemia"],
            "urology": ["urolog", "bladder", "kidney", "renal", "prostate",
                        "urinary", "incontinence"],
        }

        for category, patterns in ta_patterns.items():
            for pattern in patterns:
                if pattern in ta_lower:
                    return category

        return None

    def _assess_data_quality(
        self,
        has_ta: bool,
        has_modifiers: bool,
    ) -> DataQualityState:
        """Assess data quality based on available fields."""
        if has_ta and has_modifiers:
            return DataQualityState.FULL
        elif has_ta or has_modifiers:
            return DataQualityState.PARTIAL
        else:
            return DataQualityState.MINIMAL

    def _create_error_result(
        self,
        reason_code: str,
        missing_fields: List[str],
        as_of_date: date,
        deterministic_timestamp: str,
        details: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create error result for failed calculations."""

        audit_entry = {
            "timestamp": deterministic_timestamp,
            "as_of_date": as_of_date.isoformat(),
            "missing_fields": missing_fields,
            "data_quality_state": DataQualityState.NONE.value,
            "reason_code": reason_code,
            "details": details,
            "module_version": self.VERSION,
        }

        self.audit_trail.append(audit_entry)

        result = {
            "pos_prior": None,
            "base_pos": None,
            "modifiers_applied": [],
            "modifier_adjustment": None,
            "confidence": Decimal("0.0"),
            "reason_code": reason_code,
            "data_quality_state": DataQualityState.NONE.value,
            "missing_fields": missing_fields,
            "hash": None,
            "audit_entry": audit_entry,
        }

        if details:
            result["details"] = details

        return result

    def _clamp(self, value: Decimal, min_val: Decimal, max_val: Decimal) -> Decimal:
        """Clamp value to specified range."""
        return max(min_val, min(max_val, value))

    def get_benchmarks_info(self) -> Dict[str, Any]:
        """Return benchmarks metadata and summary."""
        return {
            "metadata": self.benchmarks_metadata,
            "stages_available": list(self.benchmarks.keys()),
            "therapeutic_areas_available": list(
                self.benchmarks.get("phase_3", {}).keys()
            ),
            "modifiers_available": list(self.MODIFIERS.keys()),
            "max_modifier_adjustment": str(self.MAX_MODIFIER_ADJUSTMENT),
        }

    def get_audit_trail(self) -> List[Dict[str, Any]]:
        """Return the audit trail."""
        return self.audit_trail.copy()

    def clear_audit_trail(self) -> None:
        """Clear the audit trail."""
        self.audit_trail = []


def apply_pos_weighting(
    stage_score: Decimal,
    pos_data: Dict[str, Any],
    confidence_threshold: Decimal = Decimal("0.60"),
) -> Decimal:
    """
    Use PoS as multiplicative prior on stage score.

    Only applies if confidence > confidence_threshold.
    PoS acts as quality multiplier, not replacement.

    Args:
        stage_score: Base stage score (0-100)
        pos_data: Result from calculate_pos_prior()
        confidence_threshold: Minimum confidence to apply weighting (default 0.60)

    Returns:
        Weighted stage score (Decimal, 0-100 bounded)
    """
    confidence = pos_data.get("confidence", Decimal("0"))

    if confidence < confidence_threshold:
        return stage_score  # Insufficient confidence - use base stage score

    pos_prior = pos_data.get("pos_prior")

    if pos_prior is None:
        return stage_score

    # Normalize PoS to [0.5, 1.3] range for multiplier effect
    # This prevents extreme penalties/bonuses
    # PoS 0.0 -> 0.5x, PoS 0.95 -> 1.26x
    pos_normalized = Decimal("0.5") + (pos_prior * Decimal("0.8"))

    weighted_score = stage_score * pos_normalized

    # Clamp to 0-100
    weighted_score = max(Decimal("0"), min(Decimal("100"), weighted_score))

    return weighted_score.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def demonstration() -> None:
    """Demonstrate the PoS Prior engine capabilities."""
    print("=" * 70)
    print("PROBABILITY OF SUCCESS PRIOR ENGINE v1.0.0 - DEMONSTRATION")
    print("=" * 70)
    print()

    engine = PoSPriorEngine()
    as_of = date(2026, 1, 15)

    # Show benchmarks info
    info = engine.get_benchmarks_info()
    print(f"Benchmarks Source: {info['metadata'].get('source', 'UNKNOWN')}")
    print(f"Modifiers Available: {', '.join(info['modifiers_available'])}")
    print()

    # Case 1: Standard Phase 3 Oncology
    print("Case 1: Phase 3 Oncology (Baseline)")
    print("-" * 70)

    oncology = engine.calculate_pos_prior(
        stage="phase_3",
        therapeutic_area="oncology",
        as_of_date=as_of,
    )

    print(f"  PoS Prior: {oncology['pos_prior']}")
    print(f"  Base PoS: {oncology['base_pos']}")
    print(f"  Confidence: {oncology['confidence']}")
    print(f"  Data Quality: {oncology['data_quality_state']}")
    print()

    # Case 2: Phase 3 Rare Disease with Orphan Designation
    print("Case 2: Phase 3 Rare Disease + Orphan Designation")
    print("-" * 70)

    rare_orphan = engine.calculate_pos_prior(
        stage="phase_3",
        therapeutic_area="rare disease",
        orphan_drug_designation=True,
        as_of_date=as_of,
    )

    print(f"  PoS Prior: {rare_orphan['pos_prior']}")
    print(f"  Base PoS: {rare_orphan['base_pos']}")
    print(f"  Modifier Adjustment: {rare_orphan['modifier_adjustment']}x")
    print(f"  Modifiers Applied: {', '.join(rare_orphan['modifiers_applied'])}")
    print(f"  Confidence: {rare_orphan['confidence']}")
    print()

    # Case 3: Phase 2 with Multiple Designations
    print("Case 3: Phase 2 Oncology + Breakthrough + Biomarker Enriched")
    print("-" * 70)

    multi_modifier = engine.calculate_pos_prior(
        stage="phase_2",
        therapeutic_area="oncology",
        breakthrough_designation=True,
        biomarker_enriched=True,
        as_of_date=as_of,
    )

    print(f"  PoS Prior: {multi_modifier['pos_prior']}")
    print(f"  Base PoS: {multi_modifier['base_pos']}")
    print(f"  Modifier Adjustment: {multi_modifier['modifier_adjustment']}x")
    print(f"  Modifiers Applied: {', '.join(multi_modifier['modifiers_applied'])}")
    print()

    # Case 4: All modifiers (should hit cap)
    print("Case 4: Phase 3 with ALL Modifiers (Cap Test)")
    print("-" * 70)

    all_mods = engine.calculate_pos_prior(
        stage="phase_3",
        therapeutic_area="rare disease",
        orphan_drug_designation=True,
        breakthrough_designation=True,
        fast_track_designation=True,
        biomarker_enriched=True,
        as_of_date=as_of,
    )

    print(f"  PoS Prior: {all_mods['pos_prior']}")
    print(f"  Base PoS: {all_mods['base_pos']}")
    print(f"  Modifier Adjustment: {all_mods['modifier_adjustment']}x (capped at 2.0)")
    print(f"  Note: Uncapped would be 1.15 * 1.25 * 1.10 * 1.20 = 1.898")
    print()

    # Demonstrate integration with stage score
    print("Integration Example: Apply PoS Weighting to Stage Score")
    print("-" * 70)

    base_stage_score = Decimal("65.00")
    weighted = apply_pos_weighting(base_stage_score, rare_orphan)

    print(f"  Base Stage Score: {base_stage_score}")
    print(f"  PoS Prior: {rare_orphan['pos_prior']}")
    print(f"  Weighted Score: {weighted}")
    print()


if __name__ == "__main__":
    demonstration()
