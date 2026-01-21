#!/usr/bin/env python3
"""
pos_engine.py

Probability of Success (PoS) Engine for Biotech Screener

Provides indication-specific PoS scoring to differentiate trial success
probability based on therapeutic area, using BIO Clinical Development
Success Rate benchmarks.

Design Philosophy (Governed Over Smart):
- DETERMINISTIC: No datetime.now() - timestamps derive from as_of_date
- STDLIB-ONLY: No external dependencies
- FAIL LOUDLY: Explicit error states, not silent defaults
- PIT DISCIPLINE: All inputs from point-in-time snapshots
- EXPLICIT CLAMPING: All scores bounded to declared ranges

Benchmarks:
- Loaded from versioned external file: data/pos_benchmarks_bio_2011_2020_v1.json
- Source: BIO Clinical Development Success Rates 2011-2020
- Values should be verified against original report before production use

Author: Wake Robin Capital Management
Version: 1.1.0
"""

import hashlib
import json
import re
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from typing import Dict, List, Optional, Any
from datetime import date
from pathlib import Path
from enum import Enum


# Module metadata
__version__ = "1.2.0"
__author__ = "Wake Robin Capital Management"


class DataQualityState(Enum):
    """Data quality classification."""
    FULL = "FULL"          # All required fields present
    PARTIAL = "PARTIAL"    # Some fields missing but scoreable
    MINIMAL = "MINIMAL"    # Minimum viable data only
    NONE = "NONE"          # Insufficient data to score


class ProbabilityOfSuccessEngine:
    """
    Indication-specific PoS scoring to differentiate clinical trial success
    probability based on therapeutic area.

    Key Design Decisions:
    - Stage score and PoS score are SEPARATE outputs (not multiplied)
    - This allows clean auditing: "high stage but low PoS" is visible
    - Composite weighting happens in the orchestrator layer

    Usage:
        engine = ProbabilityOfSuccessEngine()
        result = engine.calculate_pos_score(
            base_stage="phase_3",
            indication="oncology",
            as_of_date=date(2026, 1, 11)
        )
        print(result["pos_score"])      # 43.9 (from PoS)
        print(result["stage_score"])    # 65 (from stage)
    """

    VERSION = "1.2.0"
    BENCHMARKS_FILE = "data/pos_benchmarks_bio_2011_2020_v1.json"

    # Score ranges (explicit bounds)
    POS_SCORE_MIN = Decimal("0")
    POS_SCORE_MAX = Decimal("100")
    STAGE_SCORE_MIN = Decimal("0")
    STAGE_SCORE_MAX = Decimal("100")

    # Confidence levels
    # When stage is defaulted (unknown/empty), use low confidence so PoS
    # contributes 0 weight under gating rules (threshold typically 0.40)
    CONFIDENCE_HIGH = Decimal("0.70")      # Known stage + indication
    CONFIDENCE_MEDIUM = Decimal("0.55")    # Known stage, unknown indication
    CONFIDENCE_LOW = Decimal("0.30")       # Defaulted/unknown stage (below gating threshold)

    # Base stage scores (development phase → raw score)
    # These are SEPARATE from PoS - combined at composite layer
    STAGE_SCORES: Dict[str, Decimal] = {
        "preclinical": Decimal("10"),
        "phase_1": Decimal("20"),
        "phase_1_2": Decimal("30"),
        "phase_2": Decimal("40"),
        "phase_2_3": Decimal("52"),
        "phase_3": Decimal("65"),
        "nda_bla": Decimal("80"),
        "commercial": Decimal("90")
    }

    # Required fields for full data quality
    REQUIRED_FIELDS = ["base_stage"]
    OPTIONAL_FIELDS = ["indication", "trial_design_quality", "competitive_intensity"]

    def __init__(self, benchmarks_path: Optional[str] = None, strict: bool = False):
        """
        Initialize the PoS engine.

        Args:
            benchmarks_path: Optional path to benchmarks JSON file.
                             Defaults to data/pos_benchmarks_bio_2011_2020_v1.json
            strict: If True, raise an error if benchmark file is missing/invalid.
                    If False (default), fall back to hardcoded defaults with warning.
                    Production deployments should use strict=True.
        """
        self.benchmarks_path = benchmarks_path or self.BENCHMARKS_FILE
        self.strict = strict
        self.benchmarks: Dict[str, Any] = {}
        self.benchmarks_metadata: Dict[str, Any] = {}
        self.audit_trail: List[Dict[str, Any]] = []

        # Load benchmarks
        self._load_benchmarks()

    def _load_benchmarks(self) -> None:
        """Load PoS benchmarks from external versioned file.

        Raises:
            FileNotFoundError: If strict=True and benchmark file not found
            ValueError: If strict=True and benchmark file is invalid/corrupt
        """
        # Try relative to module, then absolute
        paths_to_try = [
            Path(__file__).parent / self.benchmarks_path,
            Path(self.benchmarks_path)
        ]

        file_found = False
        load_error = None

        for path in paths_to_try:
            if path.exists():
                file_found = True
                try:
                    with open(path, "r") as f:
                        data = json.load(f)
                    self.benchmarks_metadata = data.get("_metadata", {})
                    self.benchmarks = {
                        "phase_1": data.get("phase_1_loa", {}),
                        "phase_2": data.get("phase_2_loa", {}),
                        "phase_3": data.get("phase_3_loa", {}),
                        "nda_bla": data.get("nda_bla_loa", {})
                    }
                    return
                except json.JSONDecodeError as e:
                    load_error = ValueError(f"Invalid JSON in benchmark file {path}: {e}")
                    break
                except (OSError, IOError, KeyError, TypeError) as e:
                    load_error = ValueError(f"Error loading benchmark file {path}: {e}")
                    break

        # Handle missing or invalid benchmark file
        if self.strict:
            if load_error:
                raise load_error
            if not file_found:
                raise FileNotFoundError(
                    f"Benchmark file not found: {self.benchmarks_path}. "
                    f"Searched: {[str(p) for p in paths_to_try]}. "
                    "Use strict=False to allow fallback to hardcoded defaults."
                )

        # Non-strict mode: fallback to hardcoded defaults (with warning in metadata)
        self._use_fallback_benchmarks()

    def _use_fallback_benchmarks(self) -> None:
        """Use hardcoded fallback benchmarks when file unavailable."""
        self.benchmarks_metadata = {
            "source": "FALLBACK_HARDCODED",
            "warning": "External benchmarks file not loaded"
        }
        # Conservative fallback values
        self.benchmarks = {
            "phase_1": {"all_indications": "0.079"},
            "phase_2": {"all_indications": "0.152"},
            "phase_3": {"all_indications": "0.579"},
            "nda_bla": {"all_indications": "0.903"}
        }

    def calculate_pos_score(
        self,
        base_stage: str,
        indication: Optional[str] = None,
        trial_design_quality: Optional[Decimal] = None,
        competitive_intensity: Optional[Decimal] = None,
        as_of_date: Optional[date] = None
    ) -> Dict[str, Any]:
        """
        Calculate PoS score and stage score (SEPARATELY).

        Args:
            base_stage: Development stage ("phase_1", "phase_2", "phase_3", etc.)
            indication: Therapeutic indication (e.g., "oncology", "rare disease")
            trial_design_quality: Optional quality multiplier (0.7-1.3)
            competitive_intensity: Optional competitive penalty (0.7-1.0)
            as_of_date: Point-in-time date (REQUIRED for deterministic audit)

        Returns:
            Dict containing:
            - pos_score: Decimal 0-100 (derived from LOA probability)
            - stage_score: Decimal 0-100 (from development stage)
            - loa_probability: Decimal 0-1 (raw likelihood of approval)
            - data_quality_state: str (FULL, PARTIAL, MINIMAL, NONE)
            - missing_fields: List[str]
            - inputs_used: Dict
            - audit_entry: Dict

        Raises:
            ValueError: If as_of_date is None (required for determinism)
        """

        # Validate as_of_date (REQUIRED for determinism - no silent defaults)
        if as_of_date is None:
            raise ValueError(
                "as_of_date is required for deterministic scoring. "
                "Providing date.today() as default would violate PIT discipline."
            )

        # Deterministic timestamp from as_of_date
        deterministic_timestamp = f"{as_of_date.isoformat()}T00:00:00Z"

        # Track data quality
        missing_fields = []
        inputs_used = {}

        # Normalize and validate stage
        stage_normalized, stage_was_defaulted = self._normalize_stage(base_stage)
        inputs_used["base_stage"] = base_stage
        inputs_used["stage_normalized"] = stage_normalized
        inputs_used["stage_was_defaulted"] = stage_was_defaulted

        if stage_was_defaulted:
            missing_fields.append("base_stage")

        # Normalize indication
        indication_normalized = self._normalize_indication(indication)
        inputs_used["indication"] = indication
        inputs_used["indication_normalized"] = indication_normalized

        # Detect indication parse fallback: when indication was provided but
        # normalized to "all_indications" (the generic fallback category)
        # This is important for auditability - we should know when we couldn't
        # map an indication to a specific benchmark category
        if indication and indication_normalized == "all_indications":
            inputs_used["indication_parse_fallback"] = True

        # Track optional fields
        if trial_design_quality is not None:
            inputs_used["trial_design_quality"] = str(trial_design_quality)
        else:
            missing_fields.append("trial_design_quality")

        if competitive_intensity is not None:
            inputs_used["competitive_intensity"] = str(competitive_intensity)
        else:
            missing_fields.append("competitive_intensity")

        # Determine data quality state
        data_quality_state = self._assess_data_quality(missing_fields)

        # Calculate stage score (simple lookup)
        stage_score = self.STAGE_SCORES.get(stage_normalized, Decimal("40"))
        stage_score = self._clamp(stage_score, self.STAGE_SCORE_MIN, self.STAGE_SCORE_MAX)

        # Calculate PoS (Likelihood of Approval)
        loa_probability, provenance = self._get_loa_probability(
            stage_normalized, indication_normalized
        )

        # Convert LOA probability (0-1) to score (0-100)
        # Formula: pos_score = LOA * 100
        pos_score_raw = loa_probability * Decimal("100")

        # Apply optional adjustments to PoS score only
        adjustments_applied = {}

        if trial_design_quality is not None:
            # Clamp to declared range
            tdq_clamped = self._clamp(trial_design_quality, Decimal("0.70"), Decimal("1.30"))
            pos_score_raw = pos_score_raw * tdq_clamped
            adjustments_applied["trial_design_quality"] = str(tdq_clamped)

        if competitive_intensity is not None:
            # Clamp to declared range
            ci_clamped = self._clamp(competitive_intensity, Decimal("0.70"), Decimal("1.00"))
            pos_score_raw = pos_score_raw * ci_clamped
            adjustments_applied["competitive_intensity"] = str(ci_clamped)

        # Final clamping and rounding
        pos_score = self._clamp(pos_score_raw, self.POS_SCORE_MIN, self.POS_SCORE_MAX)
        pos_score = pos_score.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        # Calculate PoS confidence
        # Key insight: if stage was defaulted (unknown/empty), confidence should be
        # LOW so that PoS contributes 0 weight under gating rules (threshold ~0.40)
        if stage_was_defaulted:
            pos_confidence = self.CONFIDENCE_LOW  # 0.30 - below gating threshold
            confidence_reason = "stage_defaulted"
        elif indication_normalized and indication_normalized != "all_indications":
            pos_confidence = self.CONFIDENCE_HIGH  # 0.70 - full weight
            confidence_reason = "stage_and_indication_known"
        else:
            pos_confidence = self.CONFIDENCE_MEDIUM  # 0.55 - partial weight
            confidence_reason = "stage_known_indication_unknown"

        # Generate deterministic audit hash
        audit_inputs = {
            "stage": stage_normalized,
            "indication": indication_normalized,
            "as_of": as_of_date.isoformat()
        }
        inputs_hash = hashlib.sha256(
            json.dumps(audit_inputs, sort_keys=True).encode()
        ).hexdigest()[:16]

        # Audit entry (deterministic timestamp!)
        audit_entry = {
            "timestamp": deterministic_timestamp,
            "as_of_date": as_of_date.isoformat(),
            "inputs_hash": inputs_hash,
            "inputs_used": inputs_used,
            "missing_fields": missing_fields,
            "data_quality_state": data_quality_state.value,
            "calculation": {
                "stage_normalized": stage_normalized,
                "stage_was_defaulted": stage_was_defaulted,
                "indication_normalized": indication_normalized,
                "loa_probability": str(loa_probability),
                "loa_provenance": provenance,
                "pos_score_raw": str(pos_score_raw),
                "pos_score_final": str(pos_score),
                "stage_score": str(stage_score),
                "pos_confidence": str(pos_confidence),
                "confidence_reason": confidence_reason,
                "adjustments_applied": adjustments_applied
            },
            "benchmarks_source": self.benchmarks_metadata.get("source", "UNKNOWN"),
            "module_version": self.VERSION
        }

        self.audit_trail.append(audit_entry)

        return {
            "pos_score": pos_score,
            "pos_confidence": pos_confidence,
            "confidence_reason": confidence_reason,
            "stage_score": stage_score,
            "stage_was_defaulted": stage_was_defaulted,
            "loa_probability": loa_probability,
            "loa_provenance": provenance,
            "stage_normalized": stage_normalized,
            "indication_normalized": indication_normalized,
            "data_quality_state": data_quality_state.value,
            "missing_fields": missing_fields,
            "inputs_used": inputs_used,
            "audit_entry": audit_entry
        }

    def score_universe(
        self,
        universe: List[Dict[str, Any]],
        as_of_date: date
    ) -> Dict[str, Any]:
        """
        Score an entire universe of companies.

        Args:
            universe: List of dicts with keys:
                - ticker: str
                - base_stage: str (development phase)
                - indication: Optional[str]
                - trial_design_quality: Optional[Decimal]
                - competitive_intensity: Optional[Decimal]
            as_of_date: Point-in-time date (REQUIRED)

        Returns:
            Dict with scores, diagnostics, and provenance
        """

        scores = []
        indication_coverage = 0
        effective_coverage = 0  # Tickers with pos_confidence >= gating threshold
        stage_defaulted_count = 0
        stage_distribution: Dict[str, int] = {}
        confidence_distribution: Dict[str, int] = {
            "high": 0, "medium": 0, "low": 0
        }
        quality_distribution: Dict[str, int] = {
            "FULL": 0, "PARTIAL": 0, "MINIMAL": 0, "NONE": 0
        }

        # Gating threshold - below this, PoS contributes 0 weight
        GATING_THRESHOLD = Decimal("0.40")

        for company in universe:
            ticker = company.get("ticker", "UNKNOWN")

            # Convert to Decimal if needed
            tdq = self._to_decimal(company.get("trial_design_quality"))
            ci = self._to_decimal(company.get("competitive_intensity"))

            # IMPORTANT: Do NOT default base_stage here - let _normalize_stage()
            # handle missing/empty stages so it can properly flag stage_was_defaulted
            # and trigger low confidence gating
            result = self.calculate_pos_score(
                base_stage=company.get("base_stage"),  # No default - allows proper flagging
                indication=company.get("indication"),
                trial_design_quality=tdq,
                competitive_intensity=ci,
                as_of_date=as_of_date
            )

            pos_confidence = result["pos_confidence"]

            # Build flags
            flags = []
            if result["stage_was_defaulted"]:
                flags.append("stage_defaulted")
            if pos_confidence < GATING_THRESHOLD:
                flags.append("below_confidence_gate")

            scores.append({
                "ticker": ticker,
                "pos_score": result["pos_score"],
                "pos_confidence": pos_confidence,
                "confidence_reason": result["confidence_reason"],
                "stage_score": result["stage_score"],
                "stage_was_defaulted": result["stage_was_defaulted"],
                "loa_probability": result["loa_probability"],
                "loa_provenance": result["loa_provenance"],
                "stage_normalized": result["stage_normalized"],
                "indication_normalized": result["indication_normalized"],
                "data_quality_state": result["data_quality_state"],
                "missing_fields": result["missing_fields"],
                "flags": flags
            })

            # Track metrics
            if result["indication_normalized"] and result["indication_normalized"] != "all_indications":
                indication_coverage += 1

            # Track effective coverage (will actually contribute to composite)
            if pos_confidence >= GATING_THRESHOLD:
                effective_coverage += 1

            if result["stage_was_defaulted"]:
                stage_defaulted_count += 1

            # Track confidence distribution
            if pos_confidence >= self.CONFIDENCE_HIGH:
                confidence_distribution["high"] += 1
            elif pos_confidence >= self.CONFIDENCE_MEDIUM:
                confidence_distribution["medium"] += 1
            else:
                confidence_distribution["low"] += 1

            stage = result["stage_normalized"]
            stage_distribution[stage] = stage_distribution.get(stage, 0) + 1
            quality_distribution[result["data_quality_state"]] += 1

        # Deterministic content hash
        scores_json = json.dumps(
            [{"t": s["ticker"], "p": str(s["pos_score"]), "s": str(s["stage_score"])}
             for s in scores],
            sort_keys=True
        )
        content_hash = hashlib.sha256(scores_json.encode()).hexdigest()[:16]

        return {
            "as_of_date": as_of_date.isoformat(),
            "scores": scores,
            "diagnostic_counts": {
                "total_scored": len(scores),
                "indication_coverage": indication_coverage,
                "indication_coverage_pct": f"{indication_coverage / max(1, len(scores)) * 100:.1f}%",
                "effective_coverage": effective_coverage,
                "effective_coverage_pct": f"{effective_coverage / max(1, len(scores)) * 100:.1f}%",
                "stage_defaulted_count": stage_defaulted_count,
                "stage_distribution": stage_distribution,
                "confidence_distribution": confidence_distribution,
                "data_quality_distribution": quality_distribution
            },
            "provenance": {
                "module": "pos_engine",
                "module_version": self.VERSION,
                "content_hash": content_hash,
                "pit_cutoff": as_of_date.isoformat(),
                "benchmarks_source": self.benchmarks_metadata.get("source", "UNKNOWN"),
                "benchmarks_version": self.benchmarks_metadata.get("report_version", "UNKNOWN")
            }
        }

    def _get_loa_probability(
        self,
        stage: str,
        indication: Optional[str]
    ) -> tuple:
        """Get Likelihood of Approval probability from benchmarks."""

        # Map stage to benchmark key
        stage_key = stage
        if stage in ("preclinical", "phase_1_2", "phase_2_3"):
            # Interpolate for intermediate stages
            interpolation_map = {
                "preclinical": ("phase_1", Decimal("0.5")),
                "phase_1_2": ("phase_1", Decimal("1.3")),
                "phase_2_3": ("phase_2", Decimal("1.5"))
            }
            base_stage, multiplier = interpolation_map.get(stage, ("phase_2", Decimal("1.0")))
            stage_key = base_stage
        elif stage == "commercial":
            return Decimal("1.0"), "commercial_approved"
        elif stage == "nda_bla":
            stage_key = "nda_bla"

        # Get benchmark data for stage
        stage_benchmarks = self.benchmarks.get(stage_key, {})

        # Try indication-specific first
        if indication and indication in stage_benchmarks:
            loa_str = stage_benchmarks[indication]
            provenance = f"BIO_{stage_key}_{indication}"
        else:
            # Fallback to all-indications average
            loa_str = stage_benchmarks.get("all_indications", "0.10")
            provenance = f"BIO_{stage_key}_all_indications"

        try:
            loa = Decimal(loa_str)
        except (ValueError, InvalidOperation, TypeError):
            loa = Decimal("0.10")
            provenance = "fallback_parse_error"

        # Apply interpolation multiplier for intermediate stages
        if stage in ("preclinical", "phase_1_2", "phase_2_3"):
            _, multiplier = {
                "preclinical": ("phase_1", Decimal("0.5")),
                "phase_1_2": ("phase_1", Decimal("1.3")),
                "phase_2_3": ("phase_2", Decimal("1.5"))
            }.get(stage, ("phase_2", Decimal("1.0")))
            loa = loa * multiplier
            provenance = f"{provenance}_interpolated"

        # Clamp to valid range
        loa = self._clamp(loa, Decimal("0"), Decimal("1"))

        return loa, provenance

    def _normalize_stage(self, stage: str) -> tuple:
        """
        Normalize stage name to canonical format.

        Returns:
            tuple: (normalized_stage: str, was_defaulted: bool)
                   was_defaulted=True if input was empty/unknown and defaulted to phase_2
        """
        if not stage:
            return ("phase_2", True)  # Defaulted - should trigger low confidence

        stage_lower = stage.lower().strip()

        if not stage_lower:
            return ("phase_2", True)  # Defaulted - should trigger low confidence

        stage_map = {
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
            "marketed": "commercial"
        }

        if stage_lower in stage_map:
            return (stage_map[stage_lower], False)  # Valid stage found

        # Try underscore version
        stage_underscore = stage_lower.replace(" ", "_").replace("-", "_")
        if stage_underscore in stage_map:
            return (stage_map[stage_underscore], False)  # Valid stage found

        return ("phase_2", True)  # Unknown stage defaulted - should trigger low confidence

    def _normalize_indication(self, indication: Optional[str]) -> Optional[str]:
        """
        Normalize indication to benchmark categories.

        Uses word-boundary-safe matching to avoid false positives.
        Also handles category aliases from indication_mapper (cns -> neurology, etc.)
        """
        if not indication:
            return None

        indication_lower = indication.lower().strip()

        # First, handle direct category aliases from indication_mapper
        # These map mapper categories to PoS benchmark categories
        category_aliases = {
            "cns": "neurology",
            "autoimmune": "immunology",
            "gi_hepatology": "gastroenterology"
        }

        # Check if indication is a direct category name (from mapper)
        if indication_lower in category_aliases:
            return category_aliases[indication_lower]

        # Check if indication is already a valid benchmark category
        valid_categories = {
            "oncology", "rare_disease", "infectious_disease", "neurology",
            "cardiovascular", "immunology", "metabolic", "respiratory",
            "dermatology", "ophthalmology", "gastroenterology", "hematology",
            "urology"
        }
        if indication_lower in valid_categories:
            return indication_lower

        # Word-boundary patterns for each category
        # Using regex \b for word boundaries to avoid matching "os" in "dose"
        indication_patterns: Dict[str, List[str]] = {
            "oncology": [
                r"\boncology\b", r"\bcancer\b", r"\btumor\b", r"\btumour\b",
                r"\bleukemia\b", r"\blymphoma\b", r"\bcarcinoma\b", r"\bmelanoma\b",
                r"\bsarcoma\b", r"\bmyeloma\b", r"\bneoplasm\b", r"\bsolid tumor\b",
                r"\bhematologic malignancy\b", r"\bimmuno-oncology\b"
            ],
            "rare_disease": [
                r"\brare\b", r"\borphan\b", r"\bultra-rare\b", r"\bultra rare\b"
            ],
            "infectious_disease": [
                r"\binfectious\b", r"\bhiv\b", r"\bhepatitis\b", r"\bantibiotic\b",
                r"\bantiviral\b", r"\bantimicrobial\b", r"\bcovid\b", r"\bbacterial\b",
                r"\bviral\b", r"\bfungal\b", r"\bparasitic\b"
            ],
            "neurology": [
                r"\bneurology\b", r"\bcns\b", r"\balzheimer\b", r"\bparkinson\b",
                r"\bals\b", r"\bepilepsy\b", r"\bmultiple sclerosis\b",
                r"\bhuntington\b", r"\bneuropathy\b", r"\bneurodegeneration\b"
            ],
            "cardiovascular": [
                r"\bcardiovascular\b", r"\bcardio\b", r"\bheart\b", r"\bcardiac\b",
                r"\bhypertension\b", r"\barrhythmia\b", r"\batherosclerosis\b"
            ],
            "immunology": [
                r"\bimmunology\b", r"\bautoimmune\b", r"\brheumatoid\b", r"\blupus\b",
                r"\bpsoriasis\b", r"\bcrohn\b", r"\bcolitis\b", r"\binflammation\b"
            ],
            "metabolic": [
                r"\bmetabolic\b", r"\bdiabetes\b", r"\bobesity\b", r"\bnafld\b",
                r"\bnash\b", r"\blipid\b", r"\bcholesterol\b"
            ],
            "respiratory": [
                r"\brespiratory\b", r"\bcopd\b", r"\basthma\b", r"\bpulmonary\b",
                r"\bcystic fibrosis\b"
            ],
            "dermatology": [
                r"\bdermatology\b", r"\bskin\b", r"\batopic dermatitis\b",
                r"\beczema\b", r"\bacne\b", r"\bvitiligo\b"
            ],
            "ophthalmology": [
                r"\bophthalmology\b", r"\bophthalmic\b", r"\beye\b", r"\bretina\b",
                r"\bmacular\b", r"\bglaucoma\b", r"\buveitis\b"
            ],
            "gastroenterology": [
                r"\bgastroenterology\b", r"\bgastrointestinal\b",
                r"\bibd\b", r"\birritable bowel\b", r"\bliver\b", r"\bhepatic\b",
                r"\bgi_hepatology\b"
            ],
            "hematology": [
                r"\bhematology\b", r"\bblood\b", r"\bhemophilia\b", r"\bsickle cell\b",
                r"\bthalassemia\b", r"\banemia\b"
            ],
            "urology": [
                r"\burology\b", r"\burolog\b", r"\bbladder\b", r"\bkidney\b",
                r"\brenal\b", r"\bprostate\b", r"\burinary\b", r"\bincontinence\b",
                r"\berectile\b"
            ]
        }

        for category, patterns in indication_patterns.items():
            for pattern in patterns:
                if re.search(pattern, indication_lower):
                    return category

        return "all_indications"

    def _assess_data_quality(self, missing_fields: List[str]) -> DataQualityState:
        """Assess data quality based on missing fields."""
        required_missing = [f for f in missing_fields if f in self.REQUIRED_FIELDS]
        optional_missing = [f for f in missing_fields if f in self.OPTIONAL_FIELDS]

        if required_missing:
            return DataQualityState.NONE
        elif len(optional_missing) == 0:
            return DataQualityState.FULL
        elif len(optional_missing) <= 2:
            return DataQualityState.PARTIAL
        else:
            return DataQualityState.MINIMAL

    def _clamp(self, value: Decimal, min_val: Decimal, max_val: Decimal) -> Decimal:
        """Clamp value to specified range."""
        return max(min_val, min(max_val, value))

    def _to_decimal(self, value: Any) -> Optional[Decimal]:
        """Safely convert value to Decimal."""
        if value is None:
            return None
        if isinstance(value, Decimal):
            return value
        try:
            return Decimal(str(value))
        except (ValueError, InvalidOperation, TypeError):
            return None

    def get_benchmarks_info(self) -> Dict[str, Any]:
        """Return benchmarks metadata and summary."""
        return {
            "metadata": self.benchmarks_metadata,
            "stages_available": list(self.benchmarks.keys()),
            "indications_available": list(
                self.benchmarks.get("phase_3", {}).keys()
            )
        }

    def get_audit_trail(self) -> List[Dict[str, Any]]:
        """Return the audit trail."""
        return self.audit_trail.copy()

    def clear_audit_trail(self) -> None:
        """Clear the audit trail."""
        self.audit_trail = []


def demonstration() -> None:
    """Demonstrate the PoS engine capabilities."""
    print("=" * 70)
    print("PROBABILITY OF SUCCESS ENGINE v1.2.0 - DEMONSTRATION")
    print("=" * 70)
    print()

    engine = ProbabilityOfSuccessEngine()

    # Show benchmarks info
    print("Benchmarks Source:", engine.benchmarks_metadata.get("source", "UNKNOWN"))
    print()

    # Compare Phase 3 oncology vs rare disease
    print("Phase 3 Comparison: Oncology vs Rare Disease")
    print("-" * 70)

    as_of = date(2026, 1, 11)

    oncology = engine.calculate_pos_score(
        base_stage="phase_3",
        indication="oncology",
        as_of_date=as_of
    )
    rare = engine.calculate_pos_score(
        base_stage="phase_3",
        indication="rare disease",
        as_of_date=as_of
    )

    print("Oncology:")
    print(f"  PoS Score: {oncology['pos_score']}")
    print(f"  Stage Score: {oncology['stage_score']}")
    print(f"  LOA Probability: {oncology['loa_probability']}")
    print(f"  Data Quality: {oncology['data_quality_state']}")
    print()
    print("Rare Disease:")
    print(f"  PoS Score: {rare['pos_score']}")
    print(f"  Stage Score: {rare['stage_score']}")
    print(f"  LOA Probability: {rare['loa_probability']}")
    print()
    print(f"PoS Differentiation: {rare['pos_score'] - oncology['pos_score']} points")
    print()


if __name__ == "__main__":
    demonstration()
