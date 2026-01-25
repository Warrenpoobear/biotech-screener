"""
pos_ablation_framework.py - Ablation Framework for PoS Model v2

Runs controlled experiments to measure the impact of model components:
- Base-rate granularity (exact/partial/default)
- Modifier families (FDA-only, trial-only, none, both)
- Cap sensitivity (modifier cap levels)
- Confidence stratification

All analyses are deterministic and produce auditable JSON reports.

Wake Robin Capital Management - Biotech Screening System
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

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
    POS_FLOOR,
    POS_CEILING,
    DEFAULT_BASE_RATES,
)


__version__ = "1.0.0"


# =============================================================================
# ABLATION VARIANTS
# =============================================================================

class LookupMode:
    """Base rate lookup strategies for ablation."""
    EXACT_ONLY = "exact_only"
    EXACT_AND_PARTIAL = "exact_and_partial"  # Default behavior
    DEFAULT_ONLY = "default_only"


class ModifierMode:
    """Modifier family combinations for ablation."""
    NONE = "none"
    FDA_ONLY = "fda_only"
    TRIAL_ONLY = "trial_only"
    BOTH = "both"  # Default behavior


@dataclass
class AblationVariant:
    """Definition of a single ablation experiment variant."""
    name: str
    lookup_mode: str  # LookupMode
    modifier_mode: str  # ModifierMode
    modifier_cap: Optional[Decimal] = None  # None means use default

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "lookup_mode": self.lookup_mode,
            "modifier_mode": self.modifier_mode,
            "modifier_cap": str(self.modifier_cap) if self.modifier_cap else None,
        }


# Standard ablation variants
STANDARD_VARIANTS: List[AblationVariant] = [
    # Baseline (full model)
    AblationVariant("baseline", LookupMode.EXACT_AND_PARTIAL, ModifierMode.BOTH),

    # Lookup mode variants
    AblationVariant("exact_only", LookupMode.EXACT_ONLY, ModifierMode.BOTH),
    AblationVariant("default_only", LookupMode.DEFAULT_ONLY, ModifierMode.BOTH),

    # Modifier family variants
    AblationVariant("no_modifiers", LookupMode.EXACT_AND_PARTIAL, ModifierMode.NONE),
    AblationVariant("fda_only", LookupMode.EXACT_AND_PARTIAL, ModifierMode.FDA_ONLY),
    AblationVariant("trial_only", LookupMode.EXACT_AND_PARTIAL, ModifierMode.TRIAL_ONLY),

    # Cap sensitivity variants
    AblationVariant("cap_1.5", LookupMode.EXACT_AND_PARTIAL, ModifierMode.BOTH, Decimal("1.50")),
    AblationVariant("cap_2.0", LookupMode.EXACT_AND_PARTIAL, ModifierMode.BOTH, Decimal("2.00")),
    AblationVariant("cap_2.5", LookupMode.EXACT_AND_PARTIAL, ModifierMode.BOTH, Decimal("2.50")),
    AblationVariant("no_cap", LookupMode.EXACT_AND_PARTIAL, ModifierMode.BOTH, Decimal("10.00")),
]


# =============================================================================
# TEST CASES
# =============================================================================

@dataclass
class AblationTestCase:
    """A single test case for ablation analysis."""
    ticker: str
    stage: ClinicalStage
    therapeutic_area: TherapeuticArea
    mechanism_class: MechanismClass
    fda_designations: List[FDADesignation] = field(default_factory=list)
    trial_characteristics: List[TrialCharacteristic] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "stage": self.stage.value,
            "therapeutic_area": self.therapeutic_area.value,
            "mechanism_class": self.mechanism_class.value,
            "fda_designations": [d.value for d in self.fda_designations],
            "trial_characteristics": [t.value for t in self.trial_characteristics],
        }


def generate_comprehensive_test_cases() -> List[AblationTestCase]:
    """Generate comprehensive test cases covering the parameter space."""
    cases = []
    case_id = 0

    # Sample combinations across stages, TAs, and mechanisms
    test_combinations = [
        # Oncology variants
        (ClinicalStage.PHASE_3, TherapeuticArea.ONCOLOGY, MechanismClass.SMALL_MOLECULE),
        (ClinicalStage.PHASE_2, TherapeuticArea.ONCOLOGY, MechanismClass.MONOCLONAL_ANTIBODY),
        (ClinicalStage.PHASE_3, TherapeuticArea.ONCOLOGY, MechanismClass.ADC),
        (ClinicalStage.PHASE_2, TherapeuticArea.ONCOLOGY, MechanismClass.CAR_T),

        # Rare disease variants
        (ClinicalStage.PHASE_3, TherapeuticArea.RARE_DISEASE, MechanismClass.GENE_THERAPY),
        (ClinicalStage.PHASE_2, TherapeuticArea.RARE_DISEASE, MechanismClass.SMALL_MOLECULE),

        # CNS variants (challenging)
        (ClinicalStage.PHASE_3, TherapeuticArea.CNS, MechanismClass.SMALL_MOLECULE),
        (ClinicalStage.PHASE_3, TherapeuticArea.CNS, MechanismClass.MONOCLONAL_ANTIBODY),

        # Other TAs
        (ClinicalStage.PHASE_3, TherapeuticArea.CARDIOVASCULAR, MechanismClass.SMALL_MOLECULE),
        (ClinicalStage.PHASE_2, TherapeuticArea.IMMUNOLOGY, MechanismClass.MONOCLONAL_ANTIBODY),
        (ClinicalStage.PHASE_3, TherapeuticArea.INFECTIOUS_DISEASE, MechanismClass.SMALL_MOLECULE),
        (ClinicalStage.PHASE_2, TherapeuticArea.METABOLIC, MechanismClass.PEPTIDE),
        (ClinicalStage.PHASE_3, TherapeuticArea.HEMATOLOGY, MechanismClass.SMALL_MOLECULE),
        (ClinicalStage.PHASE_2, TherapeuticArea.OPHTHALMOLOGY, MechanismClass.GENE_THERAPY),

        # Edge cases - unknown combos (will use partial/default)
        (ClinicalStage.PHASE_3, TherapeuticArea.DERMATOLOGY, MechanismClass.BIOLOGIC),
        (ClinicalStage.PHASE_2, TherapeuticArea.GI, MechanismClass.RNA_THERAPEUTIC),
    ]

    # Modifier combinations
    modifier_combos = [
        ([], []),  # No modifiers
        ([FDADesignation.BREAKTHROUGH_THERAPY], []),
        ([FDADesignation.ORPHAN_DRUG], []),
        ([FDADesignation.FAST_TRACK], []),
        ([FDADesignation.BREAKTHROUGH_THERAPY, FDADesignation.ORPHAN_DRUG], []),
        ([], [TrialCharacteristic.BIOMARKER_SELECTED]),
        ([], [TrialCharacteristic.FIRST_IN_CLASS]),
        ([], [TrialCharacteristic.SURROGATE_ENDPOINT]),
        ([], [TrialCharacteristic.ACTIVE_COMPARATOR]),
        ([FDADesignation.BREAKTHROUGH_THERAPY], [TrialCharacteristic.BIOMARKER_SELECTED]),
        (
            [FDADesignation.BREAKTHROUGH_THERAPY, FDADesignation.ORPHAN_DRUG, FDADesignation.FAST_TRACK],
            [TrialCharacteristic.BIOMARKER_SELECTED, TrialCharacteristic.FIRST_IN_CLASS]
        ),  # Heavy positive modifiers
        (
            [],
            [TrialCharacteristic.SURROGATE_ENDPOINT, TrialCharacteristic.ACTIVE_COMPARATOR]
        ),  # Heavy negative modifiers
    ]

    for stage, ta, mech in test_combinations:
        for fda_mods, trial_mods in modifier_combos:
            case_id += 1
            cases.append(AblationTestCase(
                ticker=f"TEST_{case_id:04d}",
                stage=stage,
                therapeutic_area=ta,
                mechanism_class=mech,
                fda_designations=list(fda_mods),
                trial_characteristics=list(trial_mods),
            ))

    return cases


# =============================================================================
# ABLATION MODEL WRAPPER
# =============================================================================

class AblationPosModel:
    """
    Wrapper around PosModelV2 that applies ablation configurations.

    Properly ablates:
    - Lookup behavior (exact only, default only, or full fallback)
    - Modifier families (FDA only, trial only, none, or both)
    - Modifier caps (different max multiplier levels)
    """

    def __init__(
        self,
        base_model: PosModelV2,
        lookup_mode: str = LookupMode.EXACT_AND_PARTIAL,
        modifier_mode: str = ModifierMode.BOTH,
        modifier_cap: Optional[Decimal] = None,
    ):
        self.base_model = base_model
        self.lookup_mode = lookup_mode
        self.modifier_mode = modifier_mode
        self.modifier_cap = modifier_cap or base_model.max_modifier_product

        # Build ablated model configuration
        self._build_ablated_config()

    def _build_ablated_config(self):
        """Build ablated configuration based on mode settings."""
        # Adjust lookup behavior by modifying which base rates are available
        if self.lookup_mode == LookupMode.EXACT_ONLY:
            # Keep only exact matches - no partial fallback
            self._filtered_matrix = dict(self.base_model.base_rate_matrix)
            self._force_exact_only = True
        elif self.lookup_mode == LookupMode.DEFAULT_ONLY:
            # Use only stage-based defaults - empty matrix forces fallback
            self._filtered_matrix = {}
            self._force_exact_only = False
        else:
            # Default: use full matrix with fallback
            self._filtered_matrix = dict(self.base_model.base_rate_matrix)
            self._force_exact_only = False

        # Adjust modifiers based on mode
        if self.modifier_mode == ModifierMode.NONE:
            self._fda_modifiers = {}
            self._trial_modifiers = {}
        elif self.modifier_mode == ModifierMode.FDA_ONLY:
            self._fda_modifiers = dict(self.base_model.fda_modifiers)
            self._trial_modifiers = {}
        elif self.modifier_mode == ModifierMode.TRIAL_ONLY:
            self._fda_modifiers = {}
            self._trial_modifiers = dict(self.base_model.trial_modifiers)
        else:
            self._fda_modifiers = dict(self.base_model.fda_modifiers)
            self._trial_modifiers = dict(self.base_model.trial_modifiers)

    def calculate(self, test_case: AblationTestCase) -> PosCalculationResult:
        """
        Calculate PoS with ablation configuration applied.

        For EXACT_ONLY mode, we filter out test cases that would use partial
        lookups by checking if an exact match exists.
        """
        # Create ablated model instance
        ablated_model = PosModelV2(
            base_rate_matrix=self._filtered_matrix,
            fda_modifiers=self._fda_modifiers,
            trial_modifiers=self._trial_modifiers,
            pos_floor=self.base_model.pos_floor,
            pos_ceiling=self.base_model.pos_ceiling,
            max_modifier_product=self.modifier_cap,
            min_modifier_product=self.base_model.min_modifier_product,
            fixture_provenance=self.base_model.fixture_provenance,
        )

        # Apply modifier mode filter - don't pass modifiers if mode disables them
        fda_designations = test_case.fda_designations
        trial_characteristics = test_case.trial_characteristics

        if self.modifier_mode == ModifierMode.NONE:
            fda_designations = []
            trial_characteristics = []
        elif self.modifier_mode == ModifierMode.FDA_ONLY:
            trial_characteristics = []
        elif self.modifier_mode == ModifierMode.TRIAL_ONLY:
            fda_designations = []

        return ablated_model.calculate(
            ticker=test_case.ticker,
            stage=test_case.stage,
            therapeutic_area=test_case.therapeutic_area,
            mechanism_class=test_case.mechanism_class,
            fda_designations=fda_designations,
            trial_characteristics=trial_characteristics,
        )


# =============================================================================
# ABLATION ANALYSIS
# =============================================================================

@dataclass
class VariantStatistics:
    """Statistics for a single ablation variant."""
    variant_name: str
    n_cases: int
    mean_pos: Decimal
    median_pos: Decimal
    std_pos: Decimal
    min_pos: Decimal
    max_pos: Decimal
    pct_hitting_floor: Decimal
    pct_hitting_ceiling: Decimal
    lookup_distribution: Dict[str, int]
    confidence_distribution: Dict[str, int]

    def to_dict(self) -> dict:
        return {
            "variant_name": self.variant_name,
            "n_cases": self.n_cases,
            "mean_pos": str(self.mean_pos),
            "median_pos": str(self.median_pos),
            "std_pos": str(self.std_pos),
            "min_pos": str(self.min_pos),
            "max_pos": str(self.max_pos),
            "pct_hitting_floor": str(self.pct_hitting_floor),
            "pct_hitting_ceiling": str(self.pct_hitting_ceiling),
            "lookup_distribution": self.lookup_distribution,
            "confidence_distribution": self.confidence_distribution,
        }


def calculate_statistics(
    results: List[PosCalculationResult],
    variant_name: str,
    pos_floor: Decimal = POS_FLOOR,
    pos_ceiling: Decimal = POS_CEILING,
) -> VariantStatistics:
    """Calculate summary statistics for a set of results."""
    if not results:
        return VariantStatistics(
            variant_name=variant_name,
            n_cases=0,
            mean_pos=Decimal("0"),
            median_pos=Decimal("0"),
            std_pos=Decimal("0"),
            min_pos=Decimal("0"),
            max_pos=Decimal("0"),
            pct_hitting_floor=Decimal("0"),
            pct_hitting_ceiling=Decimal("0"),
            lookup_distribution={},
            confidence_distribution={},
        )

    pos_values = [r.adjusted_pos for r in results]
    n = len(pos_values)

    # Sort for median and percentiles
    sorted_pos = sorted(pos_values)

    # Mean
    mean_pos = sum(pos_values) / n

    # Median
    if n % 2 == 0:
        median_pos = (sorted_pos[n // 2 - 1] + sorted_pos[n // 2]) / 2
    else:
        median_pos = sorted_pos[n // 2]

    # Standard deviation
    variance = sum((p - mean_pos) ** 2 for p in pos_values) / n
    std_pos = variance.sqrt() if hasattr(variance, 'sqrt') else Decimal(str(variance ** Decimal("0.5")))

    # Min/Max
    min_pos = min(pos_values)
    max_pos = max(pos_values)

    # Floor/Ceiling hits
    floor_hits = sum(1 for p in pos_values if p <= pos_floor + Decimal("0.001"))
    ceiling_hits = sum(1 for p in pos_values if p >= pos_ceiling - Decimal("0.001"))

    # Lookup type distribution
    lookup_dist = defaultdict(int)
    for r in results:
        lookup_dist[r.base_pos_lookup] += 1

    # Confidence distribution
    confidence_dist = defaultdict(int)
    for r in results:
        confidence_dist[r.confidence_level] += 1

    return VariantStatistics(
        variant_name=variant_name,
        n_cases=n,
        mean_pos=mean_pos.quantize(Decimal("0.0001")),
        median_pos=median_pos.quantize(Decimal("0.0001")),
        std_pos=std_pos.quantize(Decimal("0.0001")) if isinstance(std_pos, Decimal) else Decimal(str(std_pos)).quantize(Decimal("0.0001")),
        min_pos=min_pos,
        max_pos=max_pos,
        pct_hitting_floor=Decimal(str(floor_hits * 100 / n)).quantize(Decimal("0.01")),
        pct_hitting_ceiling=Decimal(str(ceiling_hits * 100 / n)).quantize(Decimal("0.01")),
        lookup_distribution=dict(lookup_dist),
        confidence_distribution=dict(confidence_dist),
    )


# =============================================================================
# ABLATION RUNNER
# =============================================================================

@dataclass
class AblationReport:
    """Complete ablation analysis report."""
    run_id: str
    timestamp: str
    model_version: str
    framework_version: str
    fixture_provenance: Optional[Dict]
    variants: List[Dict]
    variant_statistics: List[Dict]
    pairwise_deltas: List[Dict]
    input_hash: str

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "model_version": self.model_version,
            "framework_version": self.framework_version,
            "fixture_provenance": self.fixture_provenance,
            "variants": self.variants,
            "variant_statistics": self.variant_statistics,
            "pairwise_deltas": self.pairwise_deltas,
            "input_hash": self.input_hash,
        }


def compute_input_hash(
    test_cases: List[AblationTestCase],
    variants: List[AblationVariant],
    fixture_sha: Optional[str],
) -> str:
    """Compute deterministic hash of all inputs for reproducibility."""
    content = {
        "test_cases": sorted([tc.to_dict() for tc in test_cases], key=lambda x: x["ticker"]),
        "variants": sorted([v.to_dict() for v in variants], key=lambda x: x["name"]),
        "fixture_sha256": fixture_sha,
        "framework_version": __version__,
    }
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def run_ablation_analysis(
    base_model: Optional[PosModelV2] = None,
    test_cases: Optional[List[AblationTestCase]] = None,
    variants: Optional[List[AblationVariant]] = None,
    fixture_path: Optional[Path] = None,
) -> AblationReport:
    """
    Run complete ablation analysis.

    Args:
        base_model: Pre-configured PosModelV2 (if None, uses defaults or fixture)
        test_cases: Test cases to evaluate (if None, generates comprehensive set)
        variants: Ablation variants to test (if None, uses standard variants)
        fixture_path: Path to fixture file (used if base_model is None)

    Returns:
        AblationReport with complete analysis
    """
    # Initialize model
    if base_model is None:
        if fixture_path and fixture_path.exists():
            base_model = PosModelV2.from_fixture(fixture_path)
        else:
            base_model = PosModelV2()

    # Get test cases and variants
    test_cases = test_cases or generate_comprehensive_test_cases()
    variants = variants or STANDARD_VARIANTS

    # Extract fixture provenance if available
    fixture_prov = base_model.fixture_provenance.to_dict() if base_model.fixture_provenance else None
    fixture_sha = base_model.fixture_provenance.fixture_sha256 if base_model.fixture_provenance else None

    # Compute input hash
    input_hash = compute_input_hash(test_cases, variants, fixture_sha)

    # Run all variants
    all_results: Dict[str, List[PosCalculationResult]] = {}

    for variant in variants:
        ablation_model = AblationPosModel(
            base_model=base_model,
            lookup_mode=variant.lookup_mode,
            modifier_mode=variant.modifier_mode,
            modifier_cap=variant.modifier_cap,
        )

        results = []
        for test_case in test_cases:
            result = ablation_model.calculate(test_case)
            results.append(result)

        all_results[variant.name] = results

    # Calculate statistics for each variant
    variant_stats = []
    for variant in variants:
        stats = calculate_statistics(
            all_results[variant.name],
            variant.name,
            base_model.pos_floor,
            base_model.pos_ceiling,
        )
        variant_stats.append(stats.to_dict())

    # Calculate pairwise deltas (vs baseline)
    pairwise_deltas = []
    if "baseline" in all_results:
        baseline_results = all_results["baseline"]
        for variant in variants:
            if variant.name == "baseline":
                continue

            variant_results = all_results[variant.name]
            deltas = []
            for i, (base_r, var_r) in enumerate(zip(baseline_results, variant_results)):
                delta = var_r.adjusted_pos - base_r.adjusted_pos
                deltas.append(delta)

            if deltas:
                mean_delta = sum(deltas) / len(deltas)
                abs_deltas = [abs(d) for d in deltas]
                mean_abs_delta = sum(abs_deltas) / len(abs_deltas)

                pairwise_deltas.append({
                    "variant": variant.name,
                    "vs": "baseline",
                    "mean_delta": str(mean_delta.quantize(Decimal("0.0001"))),
                    "mean_abs_delta": str(mean_abs_delta.quantize(Decimal("0.0001"))),
                    "min_delta": str(min(deltas).quantize(Decimal("0.0001"))),
                    "max_delta": str(max(deltas).quantize(Decimal("0.0001"))),
                })

    # Build report
    return AblationReport(
        run_id=input_hash[:16],
        timestamp=datetime.utcnow().isoformat(),
        model_version=base_model.fixture_provenance.fixture_version if base_model.fixture_provenance else "2.0.0",
        framework_version=__version__,
        fixture_provenance=fixture_prov,
        variants=[v.to_dict() for v in variants],
        variant_statistics=variant_stats,
        pairwise_deltas=pairwise_deltas,
        input_hash=input_hash,
    )


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Run ablation analysis and output report."""
    print("=" * 70)
    print("Wake Robin PoS Model v2.0 - Ablation Analysis")
    print("=" * 70)

    # Check for fixture file
    fixture_path = Path("pos_benchmarks_v1.json")
    if fixture_path.exists():
        print(f"\nLoading model from fixture: {fixture_path}")
    else:
        print("\nUsing default model configuration")
        fixture_path = None

    # Generate test cases
    test_cases = generate_comprehensive_test_cases()
    print(f"Generated {len(test_cases)} test cases")

    # Run analysis
    print(f"Running {len(STANDARD_VARIANTS)} ablation variants...")
    report = run_ablation_analysis(fixture_path=fixture_path)

    # Print summary
    print("\n" + "=" * 70)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 70)

    print(f"\nRun ID: {report.run_id}")
    print(f"Input Hash: {report.input_hash}")

    print("\n--- Variant Statistics ---")
    print(f"{'Variant':<20} {'Mean PoS':<12} {'Median':<12} {'Floor%':<10} {'Ceiling%':<10}")
    print("-" * 64)
    for stats in report.variant_statistics:
        print(f"{stats['variant_name']:<20} {stats['mean_pos']:<12} {stats['median_pos']:<12} {stats['pct_hitting_floor']:<10} {stats['pct_hitting_ceiling']:<10}")

    print("\n--- Pairwise Deltas (vs Baseline) ---")
    print(f"{'Variant':<20} {'Mean Δ':<12} {'Mean |Δ|':<12} {'Min Δ':<12} {'Max Δ':<12}")
    print("-" * 68)
    for delta in report.pairwise_deltas:
        print(f"{delta['variant']:<20} {delta['mean_delta']:<12} {delta['mean_abs_delta']:<12} {delta['min_delta']:<12} {delta['max_delta']:<12}")

    # Write full report to JSON
    output_path = Path("ablation_report.json")
    with open(output_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2, sort_keys=True)
    print(f"\nFull report written to: {output_path}")

    return report


if __name__ == "__main__":
    main()
