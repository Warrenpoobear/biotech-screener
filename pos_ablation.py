"""
pos_ablation.py - Ablation Testing Framework for PoS Model

Deterministic framework for measuring the impact of model components:
1. Base-rate granularity (exact / exact+partial / default-only)
2. Modifier families (FDA-only / trial-only / none / both)
3. Cap sensitivity (1.5 / 2.0 / 2.5 / uncapped)
4. Confidence stratification analysis

Outputs a single JSON report with:
- Per-variant summary statistics
- Deterministic hash for reproducibility
- Distribution analysis by confidence bucket

Wake Robin Capital Management - Biotech Screening System
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, getcontext
from pathlib import Path
from statistics import mean, median, stdev
from typing import Dict, List, Optional, Tuple, Any, Set

from pos_model_v2 import (
    PosModelV2,
    PosCalculationResult,
    ClinicalStage,
    TherapeuticArea,
    MechanismClass,
    FDADesignation,
    TrialCharacteristic,
    BaseRateEntry,
    ModifierDefinition,
    FixtureProvenance,
    FixtureLoader,
    POS_FLOOR,
    POS_CEILING,
    MAX_MODIFIER_PRODUCT,
    MIN_MODIFIER_PRODUCT,
    DEFAULT_BASE_RATES,
    BASE_RATE_MATRIX,
    FDA_DESIGNATION_MODIFIERS,
    TRIAL_CHARACTERISTIC_MODIFIERS,
)

getcontext().prec = 28


# =============================================================================
# ABLATION VARIANT DEFINITIONS
# =============================================================================

@dataclass
class AblationVariant:
    """Definition of an ablation variant configuration."""
    name: str
    description: str
    
    # Base rate configuration
    lookup_mode: str  # "exact_only", "exact_partial", "default_only"
    
    # Modifier configuration
    use_fda_modifiers: bool
    use_trial_modifiers: bool
    
    # Cap configuration
    modifier_cap: Optional[Decimal]  # None = uncapped, else cap value
    
    def get_config_hash(self) -> str:
        """Get deterministic hash of this variant configuration."""
        config = {
            "name": self.name,
            "lookup_mode": self.lookup_mode,
            "use_fda_modifiers": self.use_fda_modifiers,
            "use_trial_modifiers": self.use_trial_modifiers,
            "modifier_cap": str(self.modifier_cap) if self.modifier_cap else None,
        }
        return hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest()[:16]


# Standard ablation variants
STANDARD_VARIANTS: List[AblationVariant] = [
    # Baseline (production config)
    AblationVariant(
        name="baseline",
        description="Production configuration: exact+partial lookup, all modifiers, 2.0x cap",
        lookup_mode="exact_partial",
        use_fda_modifiers=True,
        use_trial_modifiers=True,
        modifier_cap=Decimal("2.0"),
    ),
    
    # Base-rate granularity variants
    AblationVariant(
        name="exact_only",
        description="Only use exact (stage, indication, mechanism) matches, no fallback",
        lookup_mode="exact_only",
        use_fda_modifiers=True,
        use_trial_modifiers=True,
        modifier_cap=Decimal("2.0"),
    ),
    AblationVariant(
        name="default_only",
        description="Only use stage-based defaults, ignore granular base rates",
        lookup_mode="default_only",
        use_fda_modifiers=True,
        use_trial_modifiers=True,
        modifier_cap=Decimal("2.0"),
    ),
    
    # Modifier family variants
    AblationVariant(
        name="fda_only",
        description="FDA designation modifiers only, no trial characteristics",
        lookup_mode="exact_partial",
        use_fda_modifiers=True,
        use_trial_modifiers=False,
        modifier_cap=Decimal("2.0"),
    ),
    AblationVariant(
        name="trial_only",
        description="Trial characteristic modifiers only, no FDA designations",
        lookup_mode="exact_partial",
        use_fda_modifiers=False,
        use_trial_modifiers=True,
        modifier_cap=Decimal("2.0"),
    ),
    AblationVariant(
        name="no_modifiers",
        description="No modifiers applied, pure base rate lookup",
        lookup_mode="exact_partial",
        use_fda_modifiers=False,
        use_trial_modifiers=False,
        modifier_cap=Decimal("2.0"),
    ),
    
    # Cap sensitivity variants
    AblationVariant(
        name="cap_1.5",
        description="Tighter modifier cap at 1.5x",
        lookup_mode="exact_partial",
        use_fda_modifiers=True,
        use_trial_modifiers=True,
        modifier_cap=Decimal("1.5"),
    ),
    AblationVariant(
        name="cap_2.5",
        description="Looser modifier cap at 2.5x",
        lookup_mode="exact_partial",
        use_fda_modifiers=True,
        use_trial_modifiers=True,
        modifier_cap=Decimal("2.5"),
    ),
    AblationVariant(
        name="uncapped",
        description="No modifier cap (dangerous but informative)",
        lookup_mode="exact_partial",
        use_fda_modifiers=True,
        use_trial_modifiers=True,
        modifier_cap=None,
    ),
]


# =============================================================================
# ABLATION MODEL WRAPPER
# =============================================================================

class AblationModel:
    """
    Model wrapper that applies ablation variant configuration.
    
    Modifies lookup and modifier behavior based on variant settings.
    """
    
    def __init__(
        self,
        variant: AblationVariant,
        base_rate_matrix: Optional[Dict] = None,
        fda_modifiers: Optional[Dict] = None,
        trial_modifiers: Optional[Dict] = None,
        default_stage_rates: Optional[Dict] = None,
    ):
        self.variant = variant
        self.base_rate_matrix = base_rate_matrix or BASE_RATE_MATRIX
        self.default_stage_rates = default_stage_rates or DEFAULT_BASE_RATES
        
        # Apply modifier configuration
        if variant.use_fda_modifiers:
            self.fda_modifiers = fda_modifiers or FDA_DESIGNATION_MODIFIERS
        else:
            self.fda_modifiers = {}
        
        if variant.use_trial_modifiers:
            self.trial_modifiers = trial_modifiers or TRIAL_CHARACTERISTIC_MODIFIERS
        else:
            self.trial_modifiers = {}
        
        # Set cap
        self.modifier_cap = variant.modifier_cap or Decimal("999")  # Effectively uncapped
        self.modifier_floor = MIN_MODIFIER_PRODUCT
    
    def _lookup_base_rate(
        self,
        stage: ClinicalStage,
        therapeutic_area: TherapeuticArea,
        mechanism_class: MechanismClass,
    ) -> Tuple[Decimal, str]:
        """Look up base rate with variant-specific behavior."""
        
        exact_key = (stage, therapeutic_area, mechanism_class)
        
        if self.variant.lookup_mode == "default_only":
            # Skip all granular lookups
            default_pos = self.default_stage_rates.get(stage, Decimal("0.30"))
            return default_pos, "default"
        
        # Try exact match
        if exact_key in self.base_rate_matrix:
            entry = self.base_rate_matrix[exact_key]
            return entry.base_pos, "exact"
        
        if self.variant.lookup_mode == "exact_only":
            # No fallback for exact_only mode
            default_pos = self.default_stage_rates.get(stage, Decimal("0.30"))
            return default_pos, "default"
        
        # exact_partial mode: try partial matches
        for key, entry in self.base_rate_matrix.items():
            if key[0] == stage and key[1] == therapeutic_area:
                return entry.base_pos, "partial_indication"
        
        for key, entry in self.base_rate_matrix.items():
            if key[0] == stage and key[2] == mechanism_class:
                return entry.base_pos, "partial_mechanism"
        
        default_pos = self.default_stage_rates.get(stage, Decimal("0.30"))
        return default_pos, "default"
    
    def calculate(
        self,
        ticker: str,
        stage: ClinicalStage,
        therapeutic_area: TherapeuticArea,
        mechanism_class: MechanismClass,
        fda_designations: List[FDADesignation],
        trial_characteristics: List[TrialCharacteristic],
    ) -> Dict[str, Any]:
        """Calculate PoS with ablation configuration applied."""
        
        # Step 1: Base rate lookup
        base_pos, lookup_type = self._lookup_base_rate(stage, therapeutic_area, mechanism_class)
        
        # Step 2: Calculate modifiers
        fda_product = Decimal("1.0")
        for designation in fda_designations:
            if designation in self.fda_modifiers:
                fda_product *= self.fda_modifiers[designation].factor
        
        trial_product = Decimal("1.0")
        for characteristic in trial_characteristics:
            if characteristic in self.trial_modifiers:
                trial_product *= self.trial_modifiers[characteristic].factor
        
        # Step 3: Apply cap
        total_product = fda_product * trial_product
        capped_product = max(self.modifier_floor, min(self.modifier_cap, total_product))
        
        # Step 4: Calculate adjusted PoS
        adjusted_pos = base_pos * capped_product
        adjusted_pos = max(POS_FLOOR, min(POS_CEILING, adjusted_pos))
        
        # Step 5: Determine confidence
        if lookup_type == "exact":
            confidence = "high"
        elif lookup_type in ("partial_indication", "partial_mechanism"):
            confidence = "medium"
        else:
            confidence = "low"
        
        return {
            "ticker": ticker,
            "base_pos": base_pos,
            "lookup_type": lookup_type,
            "fda_modifier": fda_product,
            "trial_modifier": trial_product,
            "total_modifier_raw": total_product,
            "total_modifier_capped": capped_product,
            "was_capped": total_product != capped_product,
            "adjusted_pos": adjusted_pos,
            "hit_floor": adjusted_pos == POS_FLOOR,
            "hit_ceiling": adjusted_pos == POS_CEILING,
            "confidence": confidence,
        }


# =============================================================================
# TEST CASE DEFINITIONS
# =============================================================================

@dataclass
class TestCase:
    """A single test case for ablation testing."""
    ticker: str
    stage: ClinicalStage
    therapeutic_area: TherapeuticArea
    mechanism_class: MechanismClass
    fda_designations: List[FDADesignation]
    trial_characteristics: List[TrialCharacteristic]
    
    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "stage": self.stage.value,
            "therapeutic_area": self.therapeutic_area.value,
            "mechanism_class": self.mechanism_class.value,
            "fda_designations": [d.value for d in self.fda_designations],
            "trial_characteristics": [t.value for t in self.trial_characteristics],
        }


def generate_standard_test_cases() -> List[TestCase]:
    """Generate a comprehensive set of test cases covering the matrix."""
    cases = []
    
    # Core clinical stages and common combos
    stage_combos = [
        (ClinicalStage.PHASE_2, TherapeuticArea.ONCOLOGY, MechanismClass.SMALL_MOLECULE),
        (ClinicalStage.PHASE_3, TherapeuticArea.ONCOLOGY, MechanismClass.SMALL_MOLECULE),
        (ClinicalStage.PHASE_3, TherapeuticArea.ONCOLOGY, MechanismClass.MONOCLONAL_ANTIBODY),
        (ClinicalStage.PHASE_3, TherapeuticArea.ONCOLOGY, MechanismClass.ADC),
        (ClinicalStage.PHASE_3, TherapeuticArea.ONCOLOGY, MechanismClass.CAR_T),
        (ClinicalStage.PHASE_2, TherapeuticArea.RARE_DISEASE, MechanismClass.GENE_THERAPY),
        (ClinicalStage.PHASE_3, TherapeuticArea.RARE_DISEASE, MechanismClass.GENE_THERAPY),
        (ClinicalStage.PHASE_3, TherapeuticArea.RARE_DISEASE, MechanismClass.SMALL_MOLECULE),
        (ClinicalStage.PHASE_2, TherapeuticArea.CNS, MechanismClass.SMALL_MOLECULE),
        (ClinicalStage.PHASE_3, TherapeuticArea.CNS, MechanismClass.SMALL_MOLECULE),
        (ClinicalStage.PHASE_3, TherapeuticArea.CNS, MechanismClass.MONOCLONAL_ANTIBODY),
        (ClinicalStage.PHASE_3, TherapeuticArea.CARDIOVASCULAR, MechanismClass.SMALL_MOLECULE),
        (ClinicalStage.PHASE_3, TherapeuticArea.IMMUNOLOGY, MechanismClass.MONOCLONAL_ANTIBODY),
        (ClinicalStage.PHASE_3, TherapeuticArea.METABOLIC, MechanismClass.PEPTIDE),
        (ClinicalStage.PHASE_3, TherapeuticArea.INFECTIOUS_DISEASE, MechanismClass.SMALL_MOLECULE),
        # Edge cases: no exact match expected
        (ClinicalStage.PHASE_3, TherapeuticArea.DERMATOLOGY, MechanismClass.BISPECIFIC),
        (ClinicalStage.PHASE_1, TherapeuticArea.RESPIRATORY, MechanismClass.RNA_THERAPEUTIC),
    ]
    
    # FDA designation combos
    fda_combos = [
        [],
        [FDADesignation.BREAKTHROUGH_THERAPY],
        [FDADesignation.ORPHAN_DRUG],
        [FDADesignation.FAST_TRACK],
        [FDADesignation.BREAKTHROUGH_THERAPY, FDADesignation.ORPHAN_DRUG],
        [FDADesignation.BREAKTHROUGH_THERAPY, FDADesignation.ORPHAN_DRUG, FDADesignation.FAST_TRACK],
    ]
    
    # Trial characteristic combos
    trial_combos = [
        [],
        [TrialCharacteristic.BIOMARKER_SELECTED],
        [TrialCharacteristic.FIRST_IN_CLASS],
        [TrialCharacteristic.SURROGATE_ENDPOINT],
        [TrialCharacteristic.ACTIVE_COMPARATOR],
        [TrialCharacteristic.BIOMARKER_SELECTED, TrialCharacteristic.FIRST_IN_CLASS],
        [TrialCharacteristic.SURROGATE_ENDPOINT, TrialCharacteristic.ACTIVE_COMPARATOR],
    ]
    
    # Generate combinations
    case_id = 0
    for stage, ta, mech in stage_combos:
        for fda in fda_combos:
            for trial in trial_combos:
                case_id += 1
                cases.append(TestCase(
                    ticker=f"TC{case_id:04d}",
                    stage=stage,
                    therapeutic_area=ta,
                    mechanism_class=mech,
                    fda_designations=fda,
                    trial_characteristics=trial,
                ))
    
    return cases


# =============================================================================
# ABLATION RUNNER
# =============================================================================

@dataclass
class VariantStats:
    """Statistics for a single ablation variant."""
    variant_name: str
    n_cases: int
    
    # PoS distribution
    mean_adjusted_pos: float
    median_adjusted_pos: float
    stdev_adjusted_pos: float
    min_adjusted_pos: float
    max_adjusted_pos: float
    
    # Floor/ceiling hits
    pct_hit_floor: float
    pct_hit_ceiling: float
    pct_was_capped: float
    
    # Lookup type distribution
    lookup_distribution: Dict[str, float]
    
    # Confidence distribution
    confidence_distribution: Dict[str, float]
    
    # Difference from baseline
    delta_mean_vs_baseline: Optional[float] = None
    
    def to_dict(self) -> dict:
        return {
            "variant_name": self.variant_name,
            "n_cases": self.n_cases,
            "pos_distribution": {
                "mean": round(self.mean_adjusted_pos, 4),
                "median": round(self.median_adjusted_pos, 4),
                "stdev": round(self.stdev_adjusted_pos, 4),
                "min": round(self.min_adjusted_pos, 4),
                "max": round(self.max_adjusted_pos, 4),
            },
            "boundary_hits": {
                "pct_hit_floor": round(self.pct_hit_floor, 2),
                "pct_hit_ceiling": round(self.pct_hit_ceiling, 2),
                "pct_was_capped": round(self.pct_was_capped, 2),
            },
            "lookup_distribution": {k: round(v, 2) for k, v in self.lookup_distribution.items()},
            "confidence_distribution": {k: round(v, 2) for k, v in self.confidence_distribution.items()},
            "delta_mean_vs_baseline": round(self.delta_mean_vs_baseline, 4) if self.delta_mean_vs_baseline else None,
        }


class AblationRunner:
    """
    Runs ablation tests across variants and generates reports.
    """
    
    def __init__(
        self,
        fixture_path: Optional[Path] = None,
        variants: Optional[List[AblationVariant]] = None,
        test_cases: Optional[List[TestCase]] = None,
    ):
        """
        Initialize ablation runner.
        
        Args:
            fixture_path: Path to fixture file (for provenance tracking)
            variants: List of variants to test (default: STANDARD_VARIANTS)
            test_cases: List of test cases (default: generated standard set)
        """
        self.variants = variants or STANDARD_VARIANTS
        self.test_cases = test_cases or generate_standard_test_cases()
        
        # Load fixture if provided
        self.fixture_provenance: Optional[FixtureProvenance] = None
        self.base_rate_matrix = BASE_RATE_MATRIX
        self.fda_modifiers = FDA_DESIGNATION_MODIFIERS
        self.trial_modifiers = TRIAL_CHARACTERISTIC_MODIFIERS
        self.default_stage_rates = DEFAULT_BASE_RATES
        
        if fixture_path:
            fixture_data = FixtureLoader.load(fixture_path)
            self.fixture_provenance = fixture_data.provenance
            self.base_rate_matrix = fixture_data.base_rate_matrix
            self.fda_modifiers = fixture_data.fda_modifiers
            self.trial_modifiers = fixture_data.trial_modifiers
            self.default_stage_rates = fixture_data.default_stage_rates
    
    def _run_variant(self, variant: AblationVariant) -> Tuple[List[Dict], VariantStats]:
        """Run all test cases through a single variant."""
        
        model = AblationModel(
            variant=variant,
            base_rate_matrix=self.base_rate_matrix,
            fda_modifiers=self.fda_modifiers,
            trial_modifiers=self.trial_modifiers,
            default_stage_rates=self.default_stage_rates,
        )
        
        results = []
        for tc in self.test_cases:
            result = model.calculate(
                ticker=tc.ticker,
                stage=tc.stage,
                therapeutic_area=tc.therapeutic_area,
                mechanism_class=tc.mechanism_class,
                fda_designations=tc.fda_designations,
                trial_characteristics=tc.trial_characteristics,
            )
            results.append(result)
        
        # Calculate statistics
        adjusted_pos_values = [float(r["adjusted_pos"]) for r in results]
        
        # Count lookup types
        lookup_counts: Dict[str, int] = defaultdict(int)
        for r in results:
            lookup_counts[r["lookup_type"]] += 1
        
        # Count confidence levels
        confidence_counts: Dict[str, int] = defaultdict(int)
        for r in results:
            confidence_counts[r["confidence"]] += 1
        
        n = len(results)
        
        stats = VariantStats(
            variant_name=variant.name,
            n_cases=n,
            mean_adjusted_pos=mean(adjusted_pos_values),
            median_adjusted_pos=median(adjusted_pos_values),
            stdev_adjusted_pos=stdev(adjusted_pos_values) if n > 1 else 0.0,
            min_adjusted_pos=min(adjusted_pos_values),
            max_adjusted_pos=max(adjusted_pos_values),
            pct_hit_floor=100 * sum(1 for r in results if r["hit_floor"]) / n,
            pct_hit_ceiling=100 * sum(1 for r in results if r["hit_ceiling"]) / n,
            pct_was_capped=100 * sum(1 for r in results if r["was_capped"]) / n,
            lookup_distribution={k: 100 * v / n for k, v in lookup_counts.items()},
            confidence_distribution={k: 100 * v / n for k, v in confidence_counts.items()},
        )
        
        return results, stats
    
    def run(self) -> "AblationReport":
        """Run all variants and generate report."""
        
        all_results: Dict[str, List[Dict]] = {}
        all_stats: Dict[str, VariantStats] = {}
        
        baseline_mean = None
        
        for variant in self.variants:
            results, stats = self._run_variant(variant)
            all_results[variant.name] = results
            all_stats[variant.name] = stats
            
            if variant.name == "baseline":
                baseline_mean = stats.mean_adjusted_pos
        
        # Calculate deltas from baseline
        if baseline_mean is not None:
            for name, stats in all_stats.items():
                if name != "baseline":
                    stats.delta_mean_vs_baseline = stats.mean_adjusted_pos - baseline_mean
        
        return AblationReport(
            runner=self,
            all_results=all_results,
            all_stats=all_stats,
        )


@dataclass
class AblationReport:
    """Complete ablation testing report."""
    runner: AblationRunner
    all_results: Dict[str, List[Dict]]
    all_stats: Dict[str, VariantStats]
    
    def get_report_hash(self) -> str:
        """Generate deterministic hash of the entire report for reproducibility."""
        hash_input = {
            "model_version": "2.1.0",
            "n_test_cases": len(self.runner.test_cases),
            "n_variants": len(self.runner.variants),
            "variant_configs": [v.get_config_hash() for v in self.runner.variants],
            "fixture_sha256": self.runner.fixture_provenance.fixture_sha256 if self.runner.fixture_provenance else None,
            "test_case_tickers": sorted([tc.ticker for tc in self.runner.test_cases]),
        }
        return hashlib.sha256(
            json.dumps(hash_input, sort_keys=True).encode()
        ).hexdigest()
    
    def to_dict(self) -> dict:
        """Export full report as dictionary."""
        return {
            "metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "report_hash": self.get_report_hash(),
                "model_version": "2.1.0",
                "n_test_cases": len(self.runner.test_cases),
                "n_variants": len(self.runner.variants),
                "fixture_provenance": self.runner.fixture_provenance.to_dict() if self.runner.fixture_provenance else None,
            },
            "variants": [
                {
                    "name": v.name,
                    "description": v.description,
                    "config_hash": v.get_config_hash(),
                    "lookup_mode": v.lookup_mode,
                    "use_fda_modifiers": v.use_fda_modifiers,
                    "use_trial_modifiers": v.use_trial_modifiers,
                    "modifier_cap": str(v.modifier_cap) if v.modifier_cap else "uncapped",
                }
                for v in self.runner.variants
            ],
            "summary_stats": {
                name: stats.to_dict()
                for name, stats in self.all_stats.items()
            },
            "comparative_analysis": self._get_comparative_analysis(),
        }
    
    def _get_comparative_analysis(self) -> dict:
        """Generate comparative analysis across variants."""
        baseline_stats = self.all_stats.get("baseline")
        if not baseline_stats:
            return {}
        
        comparisons = []
        for name, stats in self.all_stats.items():
            if name == "baseline":
                continue
            
            comparisons.append({
                "variant": name,
                "delta_mean": stats.delta_mean_vs_baseline,
                "direction": "higher" if (stats.delta_mean_vs_baseline or 0) > 0 else "lower",
                "lookup_shift": {
                    k: stats.lookup_distribution.get(k, 0) - baseline_stats.lookup_distribution.get(k, 0)
                    for k in set(stats.lookup_distribution.keys()) | set(baseline_stats.lookup_distribution.keys())
                },
                "ceiling_hit_change": stats.pct_hit_ceiling - baseline_stats.pct_hit_ceiling,
                "floor_hit_change": stats.pct_hit_floor - baseline_stats.pct_hit_floor,
            })
        
        return {
            "baseline_mean": baseline_stats.mean_adjusted_pos,
            "comparisons": comparisons,
            "highest_mean_variant": max(self.all_stats.items(), key=lambda x: x[1].mean_adjusted_pos)[0],
            "lowest_mean_variant": min(self.all_stats.items(), key=lambda x: x[1].mean_adjusted_pos)[0],
            "most_ceiling_hits": max(self.all_stats.items(), key=lambda x: x[1].pct_hit_ceiling)[0],
            "most_floor_hits": max(self.all_stats.items(), key=lambda x: x[1].pct_hit_floor)[0],
        }
    
    def to_json(self, path: Optional[Path] = None) -> str:
        """Export report as JSON string or file."""
        json_str = json.dumps(self.to_dict(), indent=2, sort_keys=True)
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(json_str)
        return json_str
    
    def print_summary(self):
        """Print human-readable summary to stdout."""
        print("=" * 70)
        print("ABLATION TEST REPORT")
        print("=" * 70)
        print(f"Report Hash: {self.get_report_hash()[:32]}...")
        print(f"Test Cases: {len(self.runner.test_cases)}")
        print(f"Variants: {len(self.runner.variants)}")
        if self.runner.fixture_provenance:
            print(f"Fixture: {self.runner.fixture_provenance.fixture_version}")
        
        print("\n" + "-" * 70)
        print("VARIANT SUMMARY STATISTICS")
        print("-" * 70)
        print(f"{'Variant':<20} {'Mean PoS':>10} {'Median':>10} {'Δ Baseline':>12} {'%Ceiling':>10} {'%Floor':>10}")
        print("-" * 70)
        
        baseline_mean = self.all_stats.get("baseline", self.all_stats[list(self.all_stats.keys())[0]]).mean_adjusted_pos
        
        for name in ["baseline", "exact_only", "default_only", "fda_only", "trial_only", 
                     "no_modifiers", "cap_1.5", "cap_2.5", "uncapped"]:
            if name not in self.all_stats:
                continue
            stats = self.all_stats[name]
            delta = stats.delta_mean_vs_baseline if stats.delta_mean_vs_baseline is not None else 0
            delta_str = f"{delta:+.4f}" if name != "baseline" else "—"
            print(f"{name:<20} {stats.mean_adjusted_pos:>10.4f} {stats.median_adjusted_pos:>10.4f} "
                  f"{delta_str:>12} {stats.pct_hit_ceiling:>9.1f}% {stats.pct_hit_floor:>9.1f}%")
        
        print("\n" + "-" * 70)
        print("LOOKUP TYPE DISTRIBUTION")
        print("-" * 70)
        print(f"{'Variant':<20} {'%Exact':>10} {'%Partial':>10} {'%Default':>10}")
        print("-" * 70)
        
        for name, stats in self.all_stats.items():
            exact = stats.lookup_distribution.get("exact", 0)
            partial = stats.lookup_distribution.get("partial_indication", 0) + \
                     stats.lookup_distribution.get("partial_mechanism", 0)
            default = stats.lookup_distribution.get("default", 0)
            print(f"{name:<20} {exact:>9.1f}% {partial:>9.1f}% {default:>9.1f}%")
        
        print("\n" + "-" * 70)
        print("CONFIDENCE DISTRIBUTION")
        print("-" * 70)
        print(f"{'Variant':<20} {'%High':>10} {'%Medium':>10} {'%Low':>10}")
        print("-" * 70)
        
        for name, stats in self.all_stats.items():
            high = stats.confidence_distribution.get("high", 0)
            medium = stats.confidence_distribution.get("medium", 0)
            low = stats.confidence_distribution.get("low", 0)
            print(f"{name:<20} {high:>9.1f}% {medium:>9.1f}% {low:>9.1f}%")
        
        print("\n" + "=" * 70)


# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

def main():
    import sys
    
    print("Wake Robin PoS Ablation Testing Framework v1.0")
    print("=" * 50)
    
    # Check for fixture path
    fixture_path = None
    output_path = None
    
    if len(sys.argv) > 1:
        fixture_path = Path(sys.argv[1])
        if not fixture_path.exists():
            print(f"ERROR: Fixture not found: {fixture_path}")
            sys.exit(1)
        print(f"Using fixture: {fixture_path}")
    
    if len(sys.argv) > 2:
        output_path = Path(sys.argv[2])
        print(f"Output path: {output_path}")
    
    # Run ablation tests
    print("\nGenerating test cases...")
    runner = AblationRunner(fixture_path=fixture_path)
    print(f"  {len(runner.test_cases)} test cases")
    print(f"  {len(runner.variants)} variants")
    
    print("\nRunning ablation tests...")
    report = runner.run()
    
    # Print summary
    report.print_summary()
    
    # Export JSON if output path provided
    if output_path:
        report.to_json(output_path)
        print(f"\nReport exported to: {output_path}")
    
    return report


if __name__ == "__main__":
    main()
